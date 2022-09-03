import numpy as np
import torch


def get_input_from_batch(params, batch, device):
    dev = torch.device(device)
    batch_size = len(batch.enc_lens)

    enc_batch = torch.from_numpy(batch.enc_batch).long().to(dev)
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float().to(dev)
    enc_lens = batch.enc_lens
    if params.use_exemplar:
        xem_batch = torch.from_numpy(batch.xem_batch).long().to(dev)
        xem_padding_mask = torch.from_numpy(batch.xem_padding_mask).float().to(dev)
        xem_lens = batch.xem_lens
        xem_similarity = torch.from_numpy(batch.xem_similarity).float().to(dev)
    else:
        xem_batch = xem_padding_mask = xem_lens = xem_similarity = None

    enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long().to(dev)
    # max_art_oovs is the max over all the article oov list in the batch
    extra_zeros = torch.zeros((batch_size, batch.max_art_oovs)).to(dev) if batch.max_art_oovs > 0 else None

    c_t_1 = torch.zeros((batch_size, 2 * params.hidden_dim)).to(dev)

    return (enc_batch, enc_padding_mask, enc_lens,
            xem_batch, xem_padding_mask, xem_lens, xem_similarity,
            enc_batch_extend_vocab, extra_zeros, c_t_1)


# noinspection PyUnusedLocal
def get_output_from_batch(params, batch, device):
    dev = torch.device(device)
    dec_batch = torch.from_numpy(batch.dec_batch).long().to(dev)
    dec_padding_mask = torch.from_numpy(batch.dec_padding_mask).float().to(dev)
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = torch.from_numpy(dec_lens).float().to(dev)

    target_batch = torch.from_numpy(batch.target_batch).long().to(dev)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
