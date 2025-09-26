

1. Jalankan `main.py` → kamera menyala, preview ditampilkan (mode preview).
2. Tekan `n` → mulai *data entry* (masukkan `nama` dan `kelas`). Program membuat buffer kosong untuk menyimpan capture sementara.
3. Tekan `SPACE` → program mendeteksi wajah di frame; jika ditemukan, crop wajah (biasanya wajah terbesar/pertama), simpan ke **buffer** (belum ke disk). Tampilkan count di layar.
4. Tekan `s` → simpan semua gambar di buffer ke folder dataset `data/<nama_kelas>/` (file .jpg), lalu kosongkan buffer.
5. Tekan `d` → hapus capture terakhir dari buffer (hanya buffer, bukan yang sudah disimpan).
6. Tekan `q` → toggle **mode deteksi saja** (preview + kotak wajah, semua operasi simpan/niat capture dinonaktifkan atau hanya preview tergantung implementasi).
7. Tekan `e` → keluar program. Jika ada gambar di buffer, program **menyimpan otomatis** (sesuai permintaanmu “keluar & semua data tersimpan”) atau menanyakan konfirmasi sebelum auto-save.

---

# Detail per tombol — apa yang terjadi internal

## `n` — buat data baru (nama + kelas)

* Prompt di terminal:

  * `Nama:` — string, misal `Andi`
  * `Kelas:` — string, misal `XIIPA`
* Sanitasi input:

  * trim spasi di depan/belakang
  * ganti spasi internal dengan underscore (`Andi_XIIPA`)
  * hapus karakter yang tidak valid untuk nama file ( `/ \ : * ? " < > |` )
* Set `current_name = "{Nama}_{Kelas}"`.
* Buat/clear buffer `captured_faces = []`.
* UI feedback: “Data baru: Andi_XIIPA — siap capture.”

## `SPACE` — capture wajah (masuk buffer)

* Ambil frame terbaru (BGR). Juga maintain grayscale versi untuk deteksi/training.
* Jalankan `face_cascade.detectMultiScale()` (Haar) atau detektor lain.
* Jika **tidak ada** wajah:

  * Tampilkan notifikasi: “No face detected — coba posisi/lampu.”
  * Jangan isi buffer.
* Jika ada >1 wajah:

  * Pilih wajah terbesar (biasa paling dekat kamera) atau indeks 0 — beri peringatan “Multiple faces — using largest”.
* Crop area wajah: `face_img = gray[y:y+h, x:x+w]`.
* (Opsional) Resize ke standar (mis. 200×200) dan simpan grayscale — ini memudahkan training LBPH:

  ```python
  face_resized = cv2.resize(face_img, (200,200))
  ```
* Append ke `captured_faces`.
* UI feedback: “Captured (1)”, gambar preview kecil (opsional).

**Catatan implementasi:** jika kamu ingin menyimpan warna aslinya, simpan BGR crop; untuk LBPH, grayscale lebih cocok.

## `s` — simpan data (dari buffer → disk)

* Validasi:

  * `current_name` harus ada. Jika tidak, minta user tekan `n`.
  * `captured_faces` tidak kosong.
* Folder target: `DATASET_DIR = "data"` → `person_dir = os.path.join(DATASET_DIR, current_name)`
  (contoh: `data/Andi_XIIPA/`)
* Jika `person_dir` tidak ada → `os.makedirs(person_dir, exist_ok=True)`.
* Simpan tiap gambar dari buffer dengan nama terurut:
  `"{current_name}_{timestamp}_{i}.jpg"`
  contoh: `Andi_XIIPA_20250926_101530_1.jpg`
* Setelah simpan, kosongkan `captured_faces`.
* Optionally, update file metadata seperti `data/users.csv` atau `label_map.json` agar training tahu label -> nama.
* UI feedback: “Saved 5 images to data/Andi_XIIPA”.

## `d` — hapus capture terakhir (buffer)

* Jika `captured_faces` kosong → “Tidak ada capture untuk dihapus”.
* Else: `captured_faces.pop()` → tampilkan count baru.
* Catatan: tombol ini hanya menghapus gambar yang **belum disimpan**. Untuk menghapus file yang telah disimpan, harus hapus manual dari folder `data/` atau sediakan fitur delete saved file.

## `q` — toggle mode deteksi wajah saja

* **Mode capture OFF**:

  * Kamera tetap menunjukkan kotak/bounding box pada wajah (preview + deteksi).
  * Menekan `SPACE` tidak menambahkan ke buffer (atau masih menambah—pilih desain). Untuk konsistensi dengan deskripsimu: di mode detect-only, `SPACE` **tidak menyimpan** — hanya preview.
  * `s` / `d` tidak melakukan apa-apa karena buffer kosong.
* UI feedback: “Mode: DETECT ONLY”
* Tekan `q` lagi untuk kembali ke mode capture.

## `e` — exit (keluar & semua data tersimpan)

* Jika `captured_faces` tidak kosong:

  * Menyimpan otomatis ke `data/<current_name>/` (sama seperti `s`) sebelum exit, sesuai permintaan “semua data tersimpan”.
  * Jika tidak ada `current_name`, program bisa:

    * mengajukan prompt: “Ada X capture belum disimpan, tekan n untuk input nama sekarang, atau 'y' untuk simpan ke folder unsaved” — atau
    * secara default menyimpan ke `data/unsaved_<timestamp>/`.
* Tutup kamera (`cap.release()`), tutup window (`cv2.destroyAllWindows()`), dan exit program.
* UI feedback: “Exit — semua data disimpan.”

---

# File & struktur penyimpanan (konkrit)

* `PROJECT_ROOT/`

  * `data/`  ← dataset (folder per-person)

    * `Andi_XIIPA/`

      * `Andi_XIIPA_20250926_101530_1.jpg`
      * `Andi_XIIPA_20250926_101532_2.jpg`
  * `models/` ← hasil training (`face_model.yml`)
  * `attendance.csv` ← log absensi (opsional)
  * `main.py`, `train_model.py`, `recognize.py`, `test_camera.py`

**Format file gambar**:

* JPEG (.jpg)
* Grayscale (recommended for LBPH) or color if kamu mau.
* Resize standar: 200×200 atau 100×100 (consistent across dataset).

---

# Setelah data terkumpul → training & mapping label

1. `train_model.py` membaca folder `data/`.
2. Membuat `label_map` (dictionary `int_id -> folder_name`), simpan sebagai `models/label_map.npy` atau `label_map.json`.
3. Latih `cv2.face.LBPHFaceRecognizer_create()` dengan gambar grayscale dan label numeric.
4. Simpan model ke `models/face_model.yml`.
5. Catatan: LBPH ringan, cocok untuk Raspberry Pi; untuk akurasi lebih tinggi bisa pakai `face_recognition` (embedding).

**Contoh mapping**:

* folder `Andi_XIIPA` → label 0
* folder `Budi_XIIPA` → label 1

---

# Recognize / Absensi (alur singkat)

* `recognize.py`:

  * load `models/face_model.yml` dan `label_map`.
  * Buka kamera → deteksi wajah setiap frame.
  * Crop → resize → `label, confidence = recognizer.predict(face_img)`
  * Tentukan threshold: (LBPH) `confidence < 80` → dianggap match; kalau >80 → Unknown.
  * Jika matched:

    * Convert label → name (pakai `label_map`).
    * Log absensi di `attendance.csv` (kolom: name, timestamp).
    * Gunakan cooldown per orang (mis. 30s atau 300s) untuk hindari duplikat pencatatan.
  * UI: Tampilkan kotak + nama + confidence.

---

# Validasi & best practices

* Ambil **minimal 10–20 foto per orang** (variasi: pose, ekspresi, pencahayaan, sudut). 3 foto sering kurang → model kurang robust.
* Pastikan **pencahayaan merata** dan wajah tidak backlight.
* Jaga **jarak kamera** konsisten (mis. 50–100 cm).
* Hindari latar belakang sibuk saat capture.
* Simpan data nama sesuai aturan privasi (hanya untuk keperluan absensi).

---

# Troubleshooting umum

* **`cv2.face` tidak ada / AttributeError**
  → install `opencv-contrib-python` (bukan `opencv-python`) dan gunakan `numpy` versi kompatibel:

  ```bash
  pip3 uninstall opencv-python opencv-contrib-python -y
  pip3 install numpy==1.26.4
  pip3 install opencv-contrib-python==4.5.5.64
  ```
* **Camera cannot open**
  → cek index (`0`, `1`), permissions, jalankan `test_camera.py`.
* **No face detected**
  → cek jarak, pencahayaan, orientasi wajah; coba gunakan `scaleFactor`/`minNeighbors` tuning pada `detectMultiScale`.
* **Saved images corrupted / cannot write**
  → cek permission folder, disk full.

---

# Rekomendasi tambahan fitur (opsional)

* Simpan `label_map.json` tiap kali `s`/`train` agar `recognize.py` mudah mapping.
* Buat `users.csv` dengan info: `id,name,class,added_at`.
* Implementasikan pilihan untuk menghapus data tersimpan dari GUI (`del_saved`).
* Tambah small GUI overlay (tkinter / PySimpleGUI) untuk input nama/keterangan tanpa harus ketik di terminal.
* Tambahkan halaman web sederhana (Flask) untuk review dataset & logs (berguna ketika remote via Tailscale + RDP).

---

# State machine ringkas (text)

* `IDLE` (preview)
  → `n` → `DATA_ENTRY`
* `DATA_ENTRY` (current_name set, buffer empty)
  → `SPACE` → append buffer (stay DATA_ENTRY)
  → `d` → pop buffer
  → `s` → save buffer → `DATA_ENTRY` (buffer cleared)
  → `q` → `DETECT_ONLY`
  → `e` → auto-save if buffer not empty → `EXIT`
* `DETECT_ONLY`
  → `q` → back to `DATA_ENTRY`/`IDLE`
  → `e` → exit

---



# Face Attendance System (Raspberry Pi + OpenCV)

Sistem absensi sederhana berbasis **Face ID** menggunakan Raspberry Pi & OpenCV.

## 📂 Struktur
- `data/` → dataset wajah per orang
- `models/` → file model hasil training
- `main.py` → aplikasi utama capture & absensi
- `train_model.py` → training dataset
- `recognize.py` → test pengenalan wajah
- `test_camera.py` → cek kamera

## 🚀 Cara Setup

### 1. Install Dependency
```bash
sudo apt update
sudo apt install python3-pip -y
pip3 install numpy==1.26.4 opencv-contrib-python==4.5.5.64

