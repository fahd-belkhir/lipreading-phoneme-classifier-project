import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from model_vit import LipReadingViTModel as LipReadingModel  # ← ViT model here
from dataset_loader import CachedTensorDataset
from train_utils import collate_fn
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Config ===
model_path = "checkpoints/model_epoch_5.pt"  # ← your ViT model checkpoint
vocab_path = "phoneme_vocab.json"
tensor_cache_dir = r"E:\lrs2_tensor_cache_val"
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
idx2phoneme = {v: k for k, v in phoneme2idx.items()}

# === Phoneme to Viseme mapping ===
phoneme_to_viseme = {
    "p": "Lip", "b": "Lip", "m": "Lip",
    "f": "Teeth", "v": "Teeth", "T": "Teeth", "D": "Teeth",
    "t": "Tongue", "d": "Tongue", "n": "Tongue", "l": "Lateral",
    "k": "Back", "g": "Back", "ŋ": "Back",
    "s": "Teeth", "z": "Teeth", "ʃ": "Teeth", "ʒ": "Teeth",
    "ʧ": "Affricate", "ʤ": "Affricate",
    "h": "Glottal",
    "r": "Glide", "j": "Glide", "w": "Glide",
    "a": "Open", "æ": "Open", "ɑ": "Back", "ɔ": "Back",
    "ʌ": "Mid", "ə": "Mid", "ɜ": "Mid",
    "e": "Front", "ɪ": "Front", "i": "Front", "ɛ": "Front",
    "u": "Round", "ʊ": "Round", "o": "Round",
    "SIL": "SIL", "<blank>": "Blank"
}

# === Greedy CTC decoder ===
def ctc_greedy_decode(log_probs, input_lengths):
    max_probs = torch.argmax(log_probs, dim=2)  # (T, B)
    results = []
    for b in range(max_probs.size(1)):
        tokens = max_probs[:input_lengths[b], b].tolist()
        collapsed = []
        prev = -1
        for t in tokens:
            if t != prev and idx2phoneme[t] != "<blank>":
                collapsed.append(t)
            prev = t
        results.append(collapsed)
    return results

# === Load model ===
model = LipReadingModel(num_classes=len(phoneme2idx)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load dataset ===
dataset = CachedTensorDataset(tensor_cache_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

# === Collect predictions ===
all_true_visemes = []
all_pred_visemes = []

print("🔍 Collecting viseme predictions...")

with torch.no_grad():
    for batch in dataloader:
        inputs, targets, input_lengths, target_lengths = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        log_probs = log_softmax(outputs, dim=2)
        pred_seqs = ctc_greedy_decode(log_probs.cpu(), input_lengths)

        start = 0
        for i, target_len in enumerate(target_lengths):
            true = targets[start:start + target_len].tolist()
            pred = pred_seqs[i]
            start += target_len

            # Handle nested lists (e.g., [[1], [2], ...])
            true = [t[0] if isinstance(t, list) else t for t in true]
            pred = [p[0] if isinstance(p, list) else p for p in pred]

            min_len = min(len(true), len(pred))
            if min_len == 0:
                continue

            true_phonemes = [idx2phoneme.get(int(t), "") for t in true[:min_len]]
            pred_phonemes = [idx2phoneme.get(int(p), "") for p in pred[:min_len]]

            true_visemes = [phoneme_to_viseme.get(p, "UNK") for p in true_phonemes]
            pred_visemes = [phoneme_to_viseme.get(p, "UNK") for p in pred_phonemes]

            all_true_visemes.extend(true_visemes)
            all_pred_visemes.extend(pred_visemes)

# === Confusion matrix ===
print("📊 Building viseme confusion matrix...")

unique_visemes = sorted(set(all_true_visemes + all_pred_visemes))
if not unique_visemes:
    print("❌ No valid viseme data found. Check your phoneme-to-viseme mapping or input tensors.")
    exit()

cm = confusion_matrix(all_true_visemes, all_pred_visemes, labels=unique_visemes, normalize="true")

fig, ax = plt.subplots(figsize=(12, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_visemes)
disp.plot(ax=ax, cmap="Purples", xticks_rotation=45, values_format=".2f")
plt.title("Confusion Matrix – ViT Model (Viseme-Level)")
plt.tight_layout()
plt.show()
