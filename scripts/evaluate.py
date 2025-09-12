import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from model import LipReadingModel
from dataset_loader import CachedTensorDataset
from train_utils import collate_fn
import json
import os

# === Config ===
model_path = "checkpoints/GRU_checkpoints/model_epoch_5.pt"
vocab_path = "phoneme_vocab.json"
tensor_cache_dir = r"E:\validation_cached"  # Validation set
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
idx2phoneme = {v: k for k, v in phoneme2idx.items()}

# === Greedy CTC decoder ===
def ctc_greedy_decode(log_probs, input_lengths):
    # log_probs: (T, B, C)
    max_probs = torch.argmax(log_probs, dim=2)  # (T, B)
    results = []
    for b in range(max_probs.size(1)):
        tokens = max_probs[:input_lengths[b], b].tolist()
        # Collapse repeats and remove blanks
        collapsed = []
        prev = -1
        for t in tokens:
            if t != prev and t != phoneme2idx["<blank>"]:
                collapsed.append(t)
            prev = t
        results.append(collapsed)
    return results

# === Manual Levenshtein distance ===
def levenshtein(pred, truth):
    m, n = len(pred), len(truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

# === Load model ===
model = LipReadingModel(num_classes=len(phoneme2idx)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Dataset and loader ===
dataset = CachedTensorDataset(tensor_cache_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

# === Evaluation loop ===
total_phonemes = 0
total_errors = 0

print("🧪 Evaluating on validation set...")

with torch.no_grad():
    for batch in dataloader:
        inputs, targets, input_lengths, target_lengths = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        outputs = model(inputs)  # (T, B, C)
        log_probs = log_softmax(outputs, dim=2)

        pred_seqs = ctc_greedy_decode(log_probs.cpu(), input_lengths.cpu())
        true_seqs = []
        start = 0
        for L in target_lengths:
            true_seqs.append(targets[start:start+L].tolist())
            start += L

        for pred, true in zip(pred_seqs, true_seqs):
            dist = levenshtein(pred, true)
            total_errors += dist
            total_phonemes += len(true)

# === Final PER ===
PER = total_errors / total_phonemes if total_phonemes > 0 else 1.0
print(f"\n✅ Phoneme Error Rate (PER): {PER:.4f}")
