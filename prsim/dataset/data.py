# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import csv
import json
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Generator, Optional, List, Tuple, Dict

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

csv.field_size_limit(2**24)


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab:
    def __init__(self, vocab_file: Path, max_size: int):
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with vocab_file.open() as vocab_f:
            # the dict of counter
            counter = Counter(json.load(vocab_f))
        for token in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            del counter[token]

        # alphabetical
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # list of (word, freq)
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for w, _ in words_and_frequencies:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            if 0 < max_size <= self._count:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                break

        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(
            self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: {}'.format(word_id))
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, path: Path):
        print("Writing word embedding metadata file to {}...".format(path))
        with path.open('w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=['word'])
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path: Path, single_pass: bool) -> Generator[Dict[str, str], None, None]:
    while True:
        with data_path.open(encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # id, abstract, article, exemplar, similarity
            yield from reader
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break


def article2ids(article_words: List[str], vocab: Vocab) -> Tuple[List[int], List[str]]:
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words: List[str], vocab: Vocab, article_oovs: List[str]) -> List[int]:
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list: List[int], vocab: Vocab, article_oovs: Optional[List[str]]) -> List[str]:
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. " \
                                             "This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            if not 0 <= article_oov_idx < len(article_oovs):  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID {} which corresponds to article OOV {}'
                                 ' but this example only has {} article OOVs'
                                 .format(i, article_oov_idx, len(article_oovs)))
            w = article_oovs[article_oov_idx]
        words.append(w)
    return words


def outputids2decwords(id_list: List[int], vocab: Vocab,
                       article_oovs: Optional[List[str]]) -> List[str]:
    decoded_words = outputids2words(id_list, vocab, article_oovs)

    # Remove the [STOP] token from decoded_words, if necessary
    with suppress(ValueError):
        fst_stop_idx = decoded_words.index(STOP_DECODING)
        del decoded_words[fst_stop_idx:]
    return decoded_words
