from dataset_loader import LipReadingDataset
from torch.utils.data import DataLoader

# === Set your paths ===
json_dir = r"E:\lrs2_aligned_json"
frames_root = r"E:\lrs2_mouth_crops"
vocab_path = r"phoneme_vocab.json"  # You are running this inside /scripts/

# === Create dataset and dataloader ===
dataset = LipReadingDataset(
    json_dir=json_dir,
    frames_root=frames_root,
    vocab_path=vocab_path
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Test one sample ===
for frames_tensor, phoneme_tensor in dataloader:
    print("Frames shape:", frames_tensor[0].shape)        # (T, C, H, W)
    print("Phoneme indices:", phoneme_tensor[0])          # tensor([12, 5, 9, ...])
    break
