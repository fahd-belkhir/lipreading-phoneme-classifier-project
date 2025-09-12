import os
import cv2
import dlib

# === Set your paths ===
frame_root = r"E:\Validation_frames"
output_root = r"E:\cropped_validation"
predictor_path = r"C:\Users\belkh\Desktop\Stage 4A\lipreading-phoneme-classifier\scripts\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

os.makedirs(output_root, exist_ok=True)

for video_folder in os.listdir(frame_root):
    video_path = os.path.join(frame_root, video_folder)
    if not os.path.isdir(video_path):
        continue

    out_video_path = os.path.join(output_root, video_folder)
    os.makedirs(out_video_path, exist_ok=True)

    for file in sorted(os.listdir(video_path)):
        if not file.endswith(".jpg"):
            continue

        img_path = os.path.join(video_path, file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"❗ No face found in {img_path}")
            continue

        landmarks = predictor(gray, faces[0])
        mouth_points = [landmarks.part(i) for i in range(48, 68)]
        x_vals = [p.x for p in mouth_points]
        y_vals = [p.y for p in mouth_points]
        x1, y1 = max(min(x_vals)-5, 0), max(min(y_vals)-5, 0)
        x2, y2 = min(max(x_vals)+5, img.shape[1]), min(max(y_vals)+5, img.shape[0])

        mouth_crop = img[y1:y2, x1:x2]
        resized_crop = cv2.resize(mouth_crop, (112, 112))

        out_path = os.path.join(out_video_path, file)
        cv2.imwrite(out_path, resized_crop)

print("✅ Finished cropping all mouth ROIs.")
