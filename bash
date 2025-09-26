# 1. Update sistem
sudo apt update && sudo apt upgrade -y

# 2. Install Python & pip
sudo apt install python3 python3-pip -y

# 3. Pastikan dependency dasar
pip3 install --upgrade pip setuptools wheel

# 4. Install versi Numpy & OpenCV yang kompatibel
pip3 uninstall -y numpy opencv-python opencv-contrib-python
pip3 install numpy==1.26.4
pip3 install opencv-contrib-python==4.5.5.64

# 5. Buat folder project
mkdir -p ~/face_attendance/{data,models}
cd ~/face_attendance

# 6. Buat file Python (nanti isi kodenya sesuai yang sudah kita buat)
nano test_camera.py
nano main.py
nano train_model.py
nano recognize.py
nano README.md

# 7. Jalankan test kamera
python3 test_camera.py

# 8. Jalankan aplikasi utama untuk input dataset
python3 main.py
# -> Tekan 'n' untuk masukkan nama_kelas
# -> Tekan 'SPACE' untuk capture wajah
# -> Tekan 's' untuk simpan
# -> Tekan 'e' untuk keluar

# 9. Training model setelah dataset terkumpul
python3 train_model.py

# 10. Jalankan pengenalan wajah (tes absensi)
python3 recognize.py
# -> Tekan 'q' untuk keluar

# 11. (Opsional) Buat requirements.txt untuk dokumentasi
echo "numpy==1.26.4
opencv-contrib-python==4.5.5.64" > requirements.txt

# 12. (Opsional) Push ke GitHub
git init
git remote add origin <URL_REPO>
git add .
git commit -m "Initial commit - Face Attendance"
git push -u origin master
