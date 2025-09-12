import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from model import LipReadingModel  # GRU model
from dataset_loader import CachedTensorDataset
from train_utils import collate_fn
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Config ===
model_path = "checkpoints/GRU_checkpoints/model_epoch_5.pt"
vocab_path = "phoneme_vocab.json"
tensor_cache_dir = r"E:\lrs2_tensor_cache_val"
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
idx2phoneme = {v: k for k, v in phoneme2idx.items()}

# === Decoder ===
def ctc_greedy_decode(log_probs, input_lengths):
    max_probs = torch.argmax(log_probs, dim=2)  # (T, B)
    results = []
    for b in range(max_probs.size(1)):
        tokens = max_probs[:input_lengths[b], b].tolist()
        collapsed = []
        prev = -1
        for t in tokens:
            if t != prev and t != phoneme2idx["<blank>"]:
                collapsed.append(t)
            prev = t
        results.append(collapsed)
    return results

# === Safe flatten function ===
def flatten(seq):
    flat = []
    for item in seq:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return [int(x) for x in flat]

# === Load model ===
model = LipReadingModel(num_classes=len(phoneme2idx)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Dataset and Loader ===
dataset = CachedTensorDataset(tensor_cache_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

# === Collect all predictions and targets ===
all_preds = []
all_trues = []

print("🔍 Collecting predictions for confusion matrix...")

with torch.no_grad():
    for batch in dataloader:
        inputs, targets, input_lengths, target_lengths = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        log_probs = log_softmax(outputs, dim=2)
        pred_seqs = ctc_greedy_decode(log_probs.cpu(), input_lengths)

        # Slice true targets
        true_seqs = []
        start = 0
        for L in target_lengths:
            true_seqs.append(targets[start:start+L].tolist())
            start += L

        for pred, true in zip(pred_seqs, true_seqs):
            pred_flat = flatten(pred)
            true_flat = flatten(true)

            if len(pred_flat) != len(true_flat):
                print(f"⚠️ Skipped sequence – length mismatch: pred={len(pred_flat)}, true={len(true_flat)}")
                continue

            all_preds.extend(pred_flat)
            all_trues.extend(true_flat)

# === Compute confusion matrix ===
print("📊 Building confusion matrix...")

labels = list(idx2phoneme.keys())
cm = confusion_matrix(all_trues, all_preds, labels=labels, normalize='true')

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[idx2phoneme[i] for i in range(len(idx2phoneme))]
)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format=".2f")
plt.title("Confusion Matrix – GRU Model")
plt.tight_layout()
plt.show()
