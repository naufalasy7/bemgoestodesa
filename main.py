import cv2
import os
import time
from ultralytics import YOLO

DATASET_DIR = "data"
os.makedirs(DATASET_DIR, exist_ok=True)

# load YOLOv11n
model = YOLO("yolov11n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera gagal dibuka")
    exit()

print("""
[INFO] Kontrol:
    n → data baru (nama)
    SPACE → capture wajah
    e → exit
""")

current_name = None
running = True

while running:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    faces = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # class 0 biasanya 'person', kita treat kepala sebagai wajah
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                faces.append(face)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        current_name = input("Masukkan nama: ").strip()
        print(f"[INFO] Data baru: {current_name}")

    elif key == ord(" "):  # spasi untuk capture
        if current_name and faces:
            save_dir = os.path.join(DATASET_DIR, current_name)
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"{int(time.time())}.jpg")
            cv2.imwrite(filename, faces[0])
            print(f"✅ Foto disimpan: {filename}")
        else:
            print("❌ Tidak ada wajah terdeteksi")

    elif key == ord("e"):
        running = False

cap.release()
cv2.destroyAllWindows()
