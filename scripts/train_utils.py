import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    frames, phonemes, input_lens, target_lens = zip(*batch)

    padded_frames = pad_sequence(frames, batch_first=True)  # (B, T, C, H, W)
    padded_phonemes = pad_sequence(phonemes, batch_first=True, padding_value=0)

    input_lens = torch.cat(input_lens, dim=0)
    target_lens = torch.cat(target_lens, dim=0)

    return padded_frames, padded_phonemes, input_lens, target_lens
