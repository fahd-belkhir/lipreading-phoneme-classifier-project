from model import LipReadingModel
import torch
import json

with open("scripts/phoneme_vocab.json", "r", encoding="utf-8") as f:
    phoneme2idx = json.load(f)

num_classes = len(phoneme2idx)

model = LipReadingModel(num_classes=num_classes)

# Dummy input: batch of 2 videos, each with 40 frames of size 112×112
dummy_input = torch.randn(2, 40, 3, 112, 112)  # (B, T, C, H, W)

output = model(dummy_input)  # (T, B, C)
print("Output shape:", output.shape)