# encoding=utf-8

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import torch
from nltk import sent_tokenize
from pydantic import BaseModel

csv.field_size_limit(2**24)


class Params(BaseModel):
    data_dir: Path
    data_file_suffix: str
    vocab_path: Path
    model_root: Path
    model_name: str = ''

    hidden_dim: int
    embed_dim: int
    batch_size: int
    max_enc_steps: int
    max_dec_steps: int
    beam_size: int
    min_dec_steps: int
    vocab_size: int

    eps: float
    max_iterations: int

    optim: str
    lr: float
    reoptim: bool
    teacher_forcing_ratio: float = 1.0

    use_exemplar: bool
    use_similarity: bool

    train_ml: bool = True
    train_rl: bool = False
    rl_weight: float = 0.0

    device: str
    eval_device: str

    summary_flush_interval: int
    print_interval: int
    eval_print_interval: int
    save_interval: int

    def save(self, json_path: Path):
        json_path.write_text(self.json(indent=4))


def use_cuda(device):
    return device and device.startswith("cuda") and torch.cuda.is_available()


def load_json_file(path: Path):
    with path.open() as f:
        return json.load(f)


def dump_json_file(path: Path, obj: object):
    with path.open('w') as f:
        json.dump(obj, f)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay={}'.format(decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return running_avg_loss


def make_html_safe(s: str) -> str:
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def replace_nl(text: str) -> str:
    return re.sub(r'\s*<nl>\s*', r'\n', text)


def prepare_rouge_text(text):
    # replace <nl> to \n
    text = replace_nl(text)
    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    text = make_html_safe(text)
    sents = sent_tokenize(text)
    text = "\n".join(sents)
    return text


def try_load_state(model_file_path: Path):
    for _ in range(3):
        try:
            return torch.load(model_file_path, map_location=lambda storage, location: storage)
        except:
            time.sleep(2)
    raise FileNotFoundError


def as_minutes(s):
    m = s // 60
    s = math.ceil(s % 60)
    return f"{m}m {s}s"


def time_since(since, percent) -> Tuple[str, str]:
    s = time.time() - since
    total = s / percent
    remain = total - s
    return as_minutes(s), as_minutes(remain)
