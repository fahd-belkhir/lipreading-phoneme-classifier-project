import whisper
import os
import json

# === Configuration ===
audio_folder = r"E:\mfa_input_val"  # ← Folder containing .wav files
output_json = "whisper_transcripts_val.json"  # ← Name of output file
model = whisper.load_model("base")  # Choose: tiny, base, small, medium, large

# === Transcription Process ===
transcripts = {}

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        wav_path = os.path.join(audio_folder, file)
        print(f"🔊 Transcribing {file}...")

        result = model.transcribe(wav_path, word_timestamps=True)
        transcripts[file] = result

# === Save All Transcripts ===
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(transcripts, f, indent=2)

print(f"\n✅ Transcriptions saved to: {output_json}")
