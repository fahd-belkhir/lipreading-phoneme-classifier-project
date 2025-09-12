import os
import subprocess
import shutil

# === SET YOUR PATHS ===
root_input = r"E:\validation_set"           # Folder with all numbered subfolders
output_folder = r"E:\mfa_input_val"       # Flattened output directory

os.makedirs(output_folder, exist_ok=True)

for subdir in os.listdir(root_input):
    subdir_path = os.path.join(root_input, subdir)
    if not os.path.isdir(subdir_path):
        continue

    for file in os.listdir(subdir_path):
        if file.endswith(".mp4"):
            base = os.path.splitext(file)[0]
            unique_id = f"{subdir}_{base}"

            mp4_path = os.path.join(subdir_path, file)
            wav_path = os.path.join(output_folder, f"{unique_id}.wav")
            txt_src_path = os.path.join(subdir_path, f"{base}.txt")
            txt_dst_path = os.path.join(output_folder, f"{unique_id}.txt")

            print(f"Extracting {unique_id}.wav...")

            subprocess.run(["ffmpeg", "-i", mp4_path, "-ar", "16000", "-ac", "1", wav_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(txt_src_path):
                shutil.copy(txt_src_path, txt_dst_path)
            else:
                print(f"❗ Transcript missing for {unique_id}")

print("✅ All .wav and .txt files extracted and renamed.")
