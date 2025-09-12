import os
import cv2

# === Set your paths ===
input_dir = r"E:\validation_set"  # Root directory with all subfolders of .mp4 files
output_dir = r"E:\Validation_frames"  # Where to save the extracted frames

frame_rate = 25  # frames per second

os.makedirs(output_dir, exist_ok=True)

for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    for file in os.listdir(subdir_path):
        if file.endswith(".mp4"):
            base = os.path.splitext(file)[0]
            video_path = os.path.join(subdir_path, file)
            output_subfolder = os.path.join(output_dir, f"{subdir}_{base}")
            os.makedirs(output_subfolder, exist_ok=True)

            print(f"Extracting frames from {file}...")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = round(fps / frame_rate) if fps > 0 else 1

            frame_idx = 0
            saved_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    filename = os.path.join(output_subfolder, f"frame_{saved_idx:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_idx += 1

                frame_idx += 1

            cap.release()

print("✅ Finished extracting frames from all videos.")
