# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py

import queue
import random
import time
from pathlib import Path
from random import shuffle
from threading import Thread
from typing import Generator, Optional, List, Tuple, Dict

import numpy as np
import tensorflow as tf

from . import data
from .data import Vocab
from ..utils import Params

random.seed(1234)
np.random.seed(318)


class Example:

    def __init__(self, params: Params, id_: str,
                 article: str, exemplar: str, similarity: float, abstract: str, vocab: Vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        del article_words[params.max_enc_steps:]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]

        if params.use_exemplar:
            # Process the exemplar
            xem_words = exemplar.split()
            del xem_words[params.max_dec_steps:]
            self.xem_len = len(xem_words)
            self.xem_input = [vocab.word2id(w) for w in xem_words]
            self.similarity = similarity

        # Process the abstract
        abstract_words = abstract.split()  # list of strings
        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, params.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
        # also store the in-article OOVs words themselves
        self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

        # Get a version of the reference summary
        # where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, params.max_dec_steps, start_decoding,
                                                    stop_decoding)

        self.params = params
        # Store the original strings
        self.id = id_
        self.original_article = article
        self.original_abstract = abstract

    @staticmethod
    def get_dec_inp_targ_seqs(
            sequence: List[int], max_len: int, start_id: int, stop_id: int) -> Tuple[List[int], List[int]]:
        # if truncate, there will be no end
        inp = [start_id] + sequence
        target = sequence + [stop_id]
        del inp[max_len:]
        del target[max_len:]
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len: int, pad_id: int):
        self.dec_input.extend([pad_id] * (max_len - len(self.dec_input)))
        self.target.extend([pad_id] * (max_len - len(self.target)))

    def pad_encoder_input(self, max_len: int, pad_id: int):
        self.enc_input.extend([pad_id] * (max_len - len(self.enc_input)))
        self.enc_input_extend_vocab.extend([pad_id] * (max_len - len(self.enc_input_extend_vocab)))

    def pad_xem_encoder_input(self, max_len: int, pad_id: int):
        self.xem_input.extend([pad_id] * (max_len - len(self.xem_input)))
        if self.xem_len < 1:
            self.xem_len = 1


class Batch:
    params: Params
    batch_size: int
    pad_id: int
    enc_batch: np.ndarray
    xem_batch: np.ndarray
    enc_lens: np.ndarray
    xem_lens: np.ndarray
    xem_similarity: np.ndarray
    enc_padding_mask: np.ndarray
    xem_padding_mask: np.ndarray
    max_art_oovs: int
    art_oovs: List[List[str]]
    enc_batch_extend_vocab: np.ndarray
    dec_batch: np.ndarray
    target_batch: np.ndarray
    dec_lens: np.ndarray
    dec_padding_mask: np.ndarray
    ids: List[str]
    original_articles: List[str]
    original_abstracts: List[str]

    def __init__(self, params: Params, example_list: List[Example], vocab: Vocab, batch_size: int):
        self.params = params
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self._init_encoder_seq(example_list)  # initialize the input to the encoder
        self._init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self._store_orig_strings(example_list)  # store the original strings

    def _init_encoder_seq(self, example_list: List[Example]):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max(ex.enc_len for ex in example_list)
        if self.params.use_exemplar:
            max_xem_len = max(1, max(ex.xem_len for ex in example_list))
        else:
            max_xem_len = 0

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
            if self.params.use_exemplar:
                ex.pad_xem_encoder_input(max_xem_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch
        # because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        # self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)
        self.enc_lens = np.zeros((self.batch_size,), dtype=np.int32)
        if self.params.use_exemplar:
            self.xem_batch = np.zeros((self.batch_size, max_xem_len), dtype=np.int32)
            self.xem_lens = np.zeros((self.batch_size,), dtype=np.int32)
            self.xem_similarity = np.zeros((self.batch_size,), dtype=np.float)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            if self.params.use_exemplar:
                self.xem_batch[i, :] = ex.xem_input[:]
                self.xem_lens[i] = ex.xem_len
                self.xem_similarity[i] = ex.similarity
        self.enc_padding_mask = (self.enc_batch != self.pad_id).astype(np.float32)
        if self.params.use_exemplar:
            self.xem_padding_mask = (self.xem_batch != self.pad_id).astype(np.float32)

        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max(len(ex.article_oovs) for ex in example_list)
        # Store the in-article OOVs themselves
        self.art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def _init_decoder_seq(self, example_list: List[Example]):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(self.params.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, self.params.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, self.params.max_dec_steps), dtype=np.int32)
        # self.dec_padding_mask = np.zeros((self.batch_size, self.params.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size,), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            # for j in range(ex.dec_len):
            #     self.dec_padding_mask[i][j] = 1
        self.dec_padding_mask = (self.dec_batch != self.pad_id).astype(np.float32)

    def _store_orig_strings(self, example_list: List[Example]):
        self.ids = [ex.id for ex in example_list]
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]


class Batcher:
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, params: Params, data_path: Path, vocab: Vocab, mode: str, batch_size: int, single_pass: bool):
        self.params = params
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue: queue.Queue[Batch] = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue: queue.Queue[Example] = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def iterate(self):
        while True:
            batch = self.next_batch()
            if batch is None:
                break
            yield batch

    def next_batch(self) -> Optional[Batch]:
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                # read the next example from file. article and abstract are both strings.
                ex_id, article, exemplar, similarity, abstract = next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if not self._single_pass:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")
                tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                self._finished_reading = True
                break

            # Process into an Example.
            example = Example(self.params, ex_id, article, exemplar, similarity, abstract, self._vocab)
            # place the Example in the example queue.
            self._example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                b = [self._example_queue.get()] * self.batch_size
                self._batch_queue.put(Batch(self.params, b, self._vocab, self.batch_size))
            else:
                # 1. add batch_size * _bucketing_cache_size examples
                # 2. sorted them in descending order
                # 3. split these examples into batches
                # 4. shuffle the order of these batches (however, in each batch, the example is still descending order)

                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = [self._example_queue.get() for _ in range(self.batch_size * self._bucketing_cache_size)]
                inputs.sort(key=lambda inp: inp.enc_len, reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(self.params, b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    @staticmethod
    def text_generator(example_generator: Generator[Dict[str, str], None, None]) \
            -> Generator[Tuple[str, str, str, float, str], None, None]:
        for e in example_generator:
            try:
                example_id = e['id']
                article_text = e['article']
                exemplar_text = e['exemplar']
                similarity = float(e['similarity'])
                abstract_text = e['abstract']
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            yield example_id, article_text, exemplar_text, similarity, abstract_text
