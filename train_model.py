import cv2
import os
import numpy as np

DATASET_DIR = "data"
MODEL_FILE = "models/face_model.yml"
os.makedirs("models", exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
current_label = 0

for person_dir in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_dir)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_dir
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(current_label)
    current_label += 1

if len(faces) == 0:
    print("❌ Tidak ada data wajah untuk training.")
    exit()

faces = np.array(faces)
labels = np.array(labels)

print(f"📂 Dataset loaded: {len(faces)} faces, {len(label_map)} persons")
recognizer.train(faces, labels)
recognizer.save(MODEL_FILE)
print(f"✅ Model trained & saved as {MODEL_FILE}")
