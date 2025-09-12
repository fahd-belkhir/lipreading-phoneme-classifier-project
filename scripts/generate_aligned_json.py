import os
import json

# === Set your paths ===
frames_root = r"E:\cropped_validation"
phn_root = r"E:\validation_output"
json_output = r"E:\Validation_aligned_json"
fps = 25

os.makedirs(json_output, exist_ok=True)

for video_id in os.listdir(frames_root):
    frames_path = os.path.join(frames_root, video_id)
    phn_path = os.path.join(phn_root, f"{video_id}.phn")

    if not os.path.isdir(frames_path) or not os.path.exists(phn_path):
        continue

    # Read phonemes
    phonemes = []
    with open(phn_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label, start, end = parts
                phonemes.append((label, float(start), float(end)))

    # Match each frame to a phoneme
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    aligned = []
    for i, file in enumerate(frame_files):
        timestamp = i / fps
        label = next((l for l, start, end in phonemes if start <= timestamp < end), "SIL")
        aligned.append((file, label))

    # Save to JSON
    output_data = {
        "frames": [f for f, _ in aligned],
        "phonemes": [p for _, p in aligned]
    }
    json_path = os.path.join(json_output, f"{video_id}.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(output_data, jf, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {json_path}")

print("🎉 All videos processed into JSON.")
