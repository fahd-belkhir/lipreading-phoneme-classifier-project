import os
import torch
import json
from tqdm import tqdm
from torchvision.io import read_image

# === Paths (customize these as needed) ===
json_dir = r"E:\lrs2_aligned_json"
frames_root = r"E:\lrs2_mouth_crops"
vocab_path = "phoneme_vocab.json"
output_dir = r"E:\Training_cached_new"

os.makedirs(output_dir, exist_ok=True)

# === Load phoneme vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    phoneme2idx = json.load(f)

# === Counters ===
total = 0
skipped_empty = 0
skipped_error = 0
cached = 0

print(f"\n🚀 Starting caching from {json_dir}...")

# === Main loop ===
for filename in tqdm(os.listdir(json_dir)):
    if not filename.endswith(".json"):
        continue

    video_id = os.path.splitext(filename)[0]
    json_path = os.path.join(json_dir, filename)
    frames_path = os.path.join(frames_root, video_id)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # === Handle phonemes ===
        phonemes_raw = data.get("phonemes", [])
        if not phonemes_raw:
            print(f"🚨 Empty phoneme list – {filename}")
            skipped_empty += 1
            continue

        if isinstance(phonemes_raw[0], dict) and "label" in phonemes_raw[0]:
            ph_labels = [p["label"] for p in phonemes_raw]
        elif isinstance(phonemes_raw[0], str):
            ph_labels = phonemes_raw
        else:
            print(f"🚨 Unknown phoneme format – {filename}")
            skipped_error += 1
            continue

        ph_tensor = torch.tensor([phoneme2idx[p] for p in ph_labels if p in phoneme2idx], dtype=torch.long)
        if ph_tensor.numel() == 0:
            print(f"🚨 No valid phonemes found in vocab – {filename}")
            skipped_empty += 1
            continue

        # === Load frames ===
        frames = []
        if not os.path.exists(frames_path):
            print(f"🚨 Frames path missing – {video_id}")
            skipped_error += 1
            continue

        frame_files = sorted([
            f for f in os.listdir(frames_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        if not frame_files:
            print(f"🚨 No frame images found – {video_id}")
            skipped_empty += 1
            continue

        for fname in frame_files:
            fpath = os.path.join(frames_path, fname)
            image = read_image(fpath) / 255.0  # Normalize to [0,1]
            frames.append(image)

        frames_tensor = torch.stack(frames)  # (T, C, H, W)

        # === Save to cache ===
        input_len = torch.tensor([frames_tensor.size(0)], dtype=torch.long)
        target_len = torch.tensor([ph_tensor.size(0)], dtype=torch.long)

        save_path = os.path.join(output_dir, f"{video_id}.pt")
        torch.save({
            'frames': frames_tensor,
            'phonemes': ph_tensor,
            'input_len': input_len,
            'target_len': target_len
        }, save_path)

        cached += 1

    except Exception as e:
        print(f"⚠️ Error processing {video_id}: {e}")
        skipped_error += 1
        continue

    total += 1

# === Summary ===
print(f"\n✅ Caching completed.")
print(f"Total JSONs scanned: {total}")
print(f"✅ Cached successfully: {cached}")
print(f"🚨 Skipped (empty): {skipped_empty}")
print(f"⚠️ Skipped (errors): {skipped_error}")
