import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from model_vit import LipReadingViTModel as LipReadingModel  # Use ViT model
from dataset_loader import CachedTensorDataset
from train_utils import collate_fn
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

# === Config ===
model_path = "checkpoints/model_epoch_5.pt"
vocab_path = "phoneme_vocab.json"
tensor_cache_dir = r"E:\lrs2_tensor_cache"  
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)
idx2phoneme = {v: k for k, v in phoneme2idx.items()}

# === Decoder ===
def ctc_greedy_decode(log_probs, input_lengths):
    max_probs = torch.argmax(log_probs, dim=2)  # (T, B, C) → (T, B)
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

# === Flatten helper ===
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

# === Load dataset ===
dataset = CachedTensorDataset(tensor_cache_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

# === Collect predictions ===
all_preds = []
all_trues = []

print("🔍 Collecting predictions for confusion matrix...")

with torch.no_grad():
    sample_index = 0
    for batch in dataloader:
        inputs, targets, input_lengths, target_lengths = batch
        inputs = inputs.to(device)

        outputs = model(inputs)
        log_probs = log_softmax(outputs, dim=2)
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long)

        pred_seqs = ctc_greedy_decode(log_probs.cpu(), input_lengths)

        start = 0
        for i, target_len in enumerate(target_lengths):
            true = targets[start:start + target_len].tolist()
            pred = pred_seqs[i]
            start += target_len

            # === DEBUG ===
            if target_len == 0 or len(true) == 0:
                print(f"🚨 Empty target – sample {sample_index}")
                sample_index += 1
                continue
            if len(pred) <= 2:
                print(f"⚠️ Very short prediction – sample {sample_index} (pred={len(pred)}, true={target_len})")

            true_flat = flatten(true)
            pred_flat = flatten(pred)

            min_len = min(len(true_flat), len(pred_flat))
            if min_len == 0:
                print(f"⚠️ Skipped – sample {sample_index}: min_len = 0")
                sample_index += 1
                continue

            all_trues.extend(true_flat[:min_len])
            all_preds.extend(pred_flat[:min_len])
            sample_index += 1

# === Confusion matrix ===
print("📊 Building confusion matrix...")

if len(all_preds) != len(all_trues):
    print(f"❌ Mismatch: preds={len(all_preds)}, trues={len(all_trues)} – truncating to shortest")
    min_len = min(len(all_preds), len(all_trues))
    all_preds = all_preds[:min_len]
    all_trues = all_trues[:min_len]

labels = list(idx2phoneme.keys())
cm = confusion_matrix(all_trues, all_preds, labels=labels, normalize='true')

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[idx2phoneme[i] for i in range(len(idx2phoneme))]
)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format=".2f")
plt.title("Confusion Matrix – ViT Model")
plt.tight_layout()
plt.show()

# === Top Phoneme Stats ===
true_phonemes = [idx2phoneme[i] for i in all_trues]
pred_phonemes = [idx2phoneme[i] for i in all_preds]

true_counts = Counter(true_phonemes)
pred_counts = Counter(pred_phonemes)

print("\n✅ Top 20 Most Frequent Phonemes in Ground Truth:")
for ph, count in true_counts.most_common(20):
    print(f"{ph:>10}: {count}")

print("\n✅ Top 20 Most Frequent Phonemes in Predictions:")
for ph, count in pred_counts.most_common(20):
    print(f"{ph:>10}: {count}")
