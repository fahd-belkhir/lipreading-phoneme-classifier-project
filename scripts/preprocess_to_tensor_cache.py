import os
import torch
from tqdm import tqdm
from dataset_loader import LipReadingDataset

# === Paths ===
json_dir = r"E:\Validation_aligned_json"
frames_root = r"E:\cropped_validation"
vocab_path = "phoneme_vocab.json"
output_dir = r"E:\validation_cached"

os.makedirs(output_dir, exist_ok=True)

# === Dataset (preprocessing mode) ===
dataset = LipReadingDataset(json_dir, frames_root, vocab_path)

print(f"\n🚀 Starting preprocessing of {len(dataset)} samples...")

for i in tqdm(range(len(dataset))):
    try:
        frames_tensor, phoneme_tensor = dataset[i]

        input_len = torch.tensor([frames_tensor.size(0)], dtype=torch.long)  # T
        target_len = torch.tensor([phoneme_tensor.size(0)], dtype=torch.long)  # L

        save_path = os.path.join(output_dir, f"sample_{i:06d}.pt")
        torch.save({
            'frames': frames_tensor,
            'phonemes': phoneme_tensor,
            'input_len': input_len,
            'target_len': target_len
        }, save_path)

    except IndexError as e:
        # These are expected (silence-heavy, empty, etc.)
        continue
    except Exception as e:
        print(f"⚠️ Error at sample {i}: {e}")
        continue

print(f"\n✅ Done. Cached tensors saved to {output_dir}")
