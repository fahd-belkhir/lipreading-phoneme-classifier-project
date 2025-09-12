import os
import json

# === Set your path ===
json_dir = r"E:\lrs2_aligned_json"

phoneme_set = set()

# Collect all unique phonemes
for file in os.listdir(json_dir):
    if not file.endswith(".json"):
        continue
    path = os.path.join(json_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        phoneme_set.update(data['phonemes'])

# Sort and add special tokens
phonemes = sorted(list(phoneme_set))
phonemes = ["<blank>"] + phonemes  # CTC blank token at index 0

# Create mapping
phoneme2idx = {phn: i for i, phn in enumerate(phonemes)}
idx2phoneme = {i: phn for phn, i in phoneme2idx.items()}

# Save to disk (optional)
import json
with open("phoneme_vocab.json", "w", encoding="utf-8") as f:
    json.dump(phoneme2idx, f, indent=2, ensure_ascii=False)

print(f"✅ Vocabulary built with {len(phonemes)} phonemes.")
print("Example mapping:", list(phoneme2idx.items())[:10])
