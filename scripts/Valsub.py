import os
import shutil
import random

# === Paths ===
full_cache_dir = r"E:\lrs2_tensor_cache"
val_cache_dir = r"E:\lrs2_tensor_cache_val"
num_val_samples = 1200  # LRS2 val set size

os.makedirs(val_cache_dir, exist_ok=True)

# === Step 1: List all .pt files
all_files = [f for f in os.listdir(full_cache_dir) if f.endswith(".pt")]
print(f"🔍 Found {len(all_files)} total .pt samples")

# === Step 2: Randomly select val subset
random.seed(42)
val_files = random.sample(all_files, min(num_val_samples, len(all_files)))

# === Step 3: Copy to validation folder
for fname in val_files:
    src = os.path.join(full_cache_dir, fname)
    dst = os.path.join(val_cache_dir, fname)
    shutil.copy(src, dst)

print(f"✅ Copied {len(val_files)} validation samples to {val_cache_dir}")
