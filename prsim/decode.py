# encoding=utf-8
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Set

import torch
from torch import Tensor

from myrouge.rouge import Rouge
from . import utils
from .dataset import data
from .dataset.batcher import Batcher, Batch
from .dataset.data import Vocab
from .dataset.train_util import get_input_from_batch
from .pointer_model import PointerEncoderDecoder
from .utils import Params, prepare_rouge_text


class Beam:
    def __init__(self,
                 tokens: List[int],
                 log_probs: List[float],
                 state: Tuple[Tensor, Tensor],  # tuple of two [hidden_size]
                 context: Tensor,  # [1 x 2*hidden_dim]
                 ngram_set: Optional[Set[Tuple[int, ...]]]):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.ngram_set = ngram_set

    def extend(self, token, log_prob, state, context, new_3gram: Tuple[int, ...]):
        if self.ngram_set is None:
            ngram_set = None
        else:
            ngram_set = copy.copy(self.ngram_set)
            ngram_set.add(new_3gram)

        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    ngram_set=ngram_set)

    def get_new_3gram(self, token: int) -> Tuple[int, ...]:
        new_3gram = tuple(self.tokens[-2:] + [token])
        return new_3gram

    def is_dup_3gram(self, new_3gram: Tuple[int, ...]) -> bool:
        return new_3gram in self.ngram_set

    @property
    def latest_token(self) -> int:
        return self.tokens[-1]

    @property
    def avg_log_prob(self) -> float:
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch:
    def __init__(self, params: Params, model_file_path: Path,
                 data_file_prefix: str = "test.", ngram_filter: bool = False):
        if data_file_prefix != "test." and ngram_filter:
            print("Warning: Using ngram_filter when validating!")
        model_name = model_file_path.name
        self.decode_dir = model_file_path.parent / '{}decode_{}'.format(data_file_prefix, model_name)
        self.decode_dir.mkdir(parents=True, exist_ok=True)

        self.vocab = Vocab(params.vocab_path, params.vocab_size)
        decode_data_path = params.data_dir / (data_file_prefix + params.data_file_suffix)
        self.batcher = Batcher(params, decode_data_path, self.vocab, mode='decode',
                               batch_size=params.beam_size, single_pass=True)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        assert (self.pad_id == 1)
        time.sleep(10)

        self.model = PointerEncoderDecoder(params, model_file_path, pad_id=self.pad_id, is_eval=True)
        self.params = params

        self.ngram_filter = ngram_filter
        if not self.ngram_filter:
            self.cand_beam_size = self.params.beam_size * 2
        else:
            self.cand_beam_size = self.params.beam_size * 5

    @staticmethod
    def sort_beams(beams: List[Beam]) -> List[Beam]:
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        start = time.time()
        counter = 0
        decode_fn = self.decode_dir / 'decode.csv'
        with decode_fn.open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, 'id ref hyp len LogProb'.split())
            writer.writeheader()

            for counter, batch in enumerate(self.batcher.iterate(), start=1):
                # Run beam search to get best Hypothesis
                try:
                    best_summary = self.beam_search(batch)
                except IndexError:
                    continue

                # Extract the output ids from the hypothesis and convert back to words
                output_ids = [int(t) for t in best_summary.tokens[1:]]
                article_oovs = batch.art_oovs[0]
                decoded_words = data.outputids2decwords(output_ids, self.vocab, article_oovs)

                # there four duplicate examples, so we just need one of them
                id_ = batch.ids[0]
                original_abstract = batch.original_abstracts[0]

                writer.writerow({
                    'id': id_,
                    'ref': original_abstract,
                    'hyp': ' '.join(decoded_words),
                    'len': len(best_summary.log_probs) - 1,
                    'LogProb': sum(best_summary.log_probs),
                })

                if counter % self.params.eval_print_interval == 0:
                    print('%d example in %d sec' % (counter, time.time() - start))
                    sys.stdout.flush()
                    start = time.time()

        print("Decoder has finished reading dataset for single_pass.")

        rouge = Rouge()
        results_raw = []
        metrics = [y + z for y in '12l' for z in 'fpr']
        with decode_fn.open(encoding='utf8', newline='') as f:
            refs = []
            hyps = []
            for item in csv.DictReader(f):
                ref, hyp = item['ref'], item['hyp']

                hyp_rouge = prepare_rouge_text(hyp)
                if not hyp_rouge:
                    continue
                [item_result] = rouge.get_scores(hyp_rouge, prepare_rouge_text(ref))

                refs.append(ref)
                hyps.append(hyp)
                item_result['LogProb'] = item['LogProb']
                item_result['len'] = item['len']
                results_raw.append(item_result)

        result_dict = {f'rouge_{y}_{z}_score': sum(it[f'rouge-{y}'][z] for it in results_raw) / counter
                       for y, z in metrics}

        print("Scores of python rouge:")
        print(result_dict)
        utils.dump_json_file(self.decode_dir / 'ROUGE_results.json', result_dict)

        with (self.decode_dir / 'scores.csv').open('w', newline='') as f:
            header = ['LogConf', 'Len'] + ['R' + x.upper() for x in metrics]
            writer = csv.DictWriter(f, header)
            writer.writeheader()
            for res in results_raw:
                item = {f'R{y}{z}'.upper(): res[f'rouge-{y}'][z]
                        for y, z in metrics}
                item['LogConf'] = res['LogProb']
                item['Len'] = res['len']
                writer.writerow(item)
        return result_dict

    def beam_search(self, batch: Batch) -> Beam:
        device = torch.device(self.params.eval_device)

        (enc_batch, enc_padding_mask, enc_lens,
         xem_batch, xem_padding_mask, xem_lens, xem_similarity,
         enc_batch_extended, extend_vocab_zeros, c_t_1
         ) = get_input_from_batch(self.params, batch, self.params.eval_device)
        # c_t_1: batch_size x 1 x 2*hidden_dim
        c_t_1 = c_t_1.unsqueeze(1)

        # enc_outputs: batch_size x max_seq_len x 2*hidden_dim
        # enc_features: batch_size x max_seq_len x 2*hidden_dim
        # dec_h: 1 x batch_size x hidden_size
        # dec_c: 1 x batch_size x hidden_size
        enc_outputs, enc_features, xem_outputs, xem_features, (dec_h, dec_c) = (
            self.model.encoder(enc_batch, enc_lens, xem_batch, xem_lens, xem_similarity))

        # batch_size x hidden_size
        dec_h: Tensor = dec_h.squeeze()
        dec_c: Tensor = dec_c.squeeze()

        # initialize beams
        # TODO: maybe we only need one beam since only beams[0] will be used later at step 0
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_1[0],
                      ngram_set=set() if self.ngram_filter else None)
                 for _ in range(self.params.beam_size)]
        results = []
        steps = 0
        while steps < self.params.max_dec_steps and len(results) < self.params.beam_size and steps < enc_lens.max():
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN)
                             for t in latest_tokens]
            y_t_1: Tensor = torch.LongTensor(latest_tokens).to(device)  # [batch_size x 1]
            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1: Tuple[Tensor, Tensor] = (torch.stack(all_state_h, 0).unsqueeze(0),  # [1 x batch_size x hidden_dim]
                                            torch.stack(all_state_c, 0).unsqueeze(0))  # [1 x batch_size x hidden_dim]
            c_t_1 = torch.stack(all_context, 0)  # [batch_size x 1 x 2*hidden_dim]

            final_dist, s_t, c_t, attn_dist = self.model.decoder(
                y_t_1, s_t_1, c_t_1, enc_outputs, enc_features, enc_padding_mask,
                xem_outputs, xem_features, xem_padding_mask, extend_vocab_zeros, enc_batch_extended)

            log_probs = torch.log(final_dist)
            # for debug
            if torch.isnan(log_probs).any():
                print("Error: log probs contains NAN!")

            topk_log_probs, topk_ids = torch.topk(log_probs, self.cand_beam_size)

            dec_h, dec_c = s_t
            # batch_size x hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            for i, h in enumerate(beams if steps != 0 else beams[:1]):
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]

                cur_count = 0
                # we assume that all beam can get no_dup 3-grams in self.cand_beam_size
                for j in range(self.cand_beam_size):  # for each of the top can_beam_size hyps:
                    cur_token = topk_ids[i, j].item()
                    new_3gram = None
                    if self.ngram_filter:
                        new_3gram = h.get_new_3gram(cur_token)
                        if h.is_dup_3gram(new_3gram):
                            continue

                    new_beam = h.extend(token=topk_ids[i, j].item(), log_prob=topk_log_probs[i, j].item(),
                                        state=state_i, context=context_i, new_3gram=new_3gram)
                    all_beams.append(new_beam)
                    cur_count += 1
                    if cur_count == self.params.beam_size:
                        break

            if len(all_beams) < 4:
                print("Error: Only find {} candidate beams.".format(all_beams))

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= self.params.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.params.beam_size or len(results) == self.params.beam_size:
                    break
            if not beams:
                break
            if len(beams) < self.params.beam_size:
                beams.extend([beams[-1]] * (self.params.beam_size - len(beams)))

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]
