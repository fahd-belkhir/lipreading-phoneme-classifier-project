import torch

# Load your cached tensor file
path = r"E:\Training_cached_new\5535415699068794046_00001.pt"  # or any other .pt path
data = torch.load(path)

video_tensor, phoneme_tensor = data['video'], data['phonemes']

print("🎥 Video tensor shape:", video_tensor.shape)
print("🗣️  Phoneme tensor shape:", phoneme_tensor.shape)
print("🗣️  Phoneme indices:", phoneme_tensor.tolist())
