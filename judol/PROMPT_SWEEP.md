# Prompt Claude Code — Sweep Otonom Stratified 5-Fold, 3 Dataset Judol

Salin-tempel prompt di bawah ini ke Claude Code (di repo `~/ultralytics`):

---

Jalankan sweep training otonom untuk SEMUA model eksperimen di repo ini pada 3 dataset Judol,
dengan stratified 5-fold cross-validation, 100 epoch. Kerjakan sepenuhnya otonom — jangan
berhenti untuk bertanya; kalau ada error, perbaiki sendiri lalu lanjutkan loop.

## Setup
1. Dataset: `judol/dataset_judi_online_yolo` (nc=2), `judol/Judol-Detection-v2-1` (nc=5),
   `judol/Judol-Detection-v2-9` (nc=5).
2. Kalau folder `judol/<ds>_strat5fold/` belum ada, jalankan `python judol/make_stratified_folds.py`
   (stratifikasi per kelas dominan tiap gambar, seed 0, symlink). Kalau skrip itu belum ada,
   buat: gabungkan train+valid+test, hitung kelas dominan tiap file label YOLO
   (`-1` untuk background), pakai `sklearn.StratifiedKFold(5, shuffle=True, random_state=0)`,
   tulis `fold_k/{train,valid}/{images,labels}` sebagai symlink + `data.yaml` ber-path absolut.

## Eksekusi
3. Jalankan `PYTHONPATH=$PWD nohup python judol/run_sweep.py > judol_sweep/sweep.log 2>&1 &`.
   Kalau skrip belum ada, buat dengan spesifikasi:
   - Antrian = semua yaml eksperimen di `ultralytics/cfg/models/{v8,26,11,12}/` (exclude
     varian -seg/-pose/-obb/-cls/-world/p2/p6/rtdetr/yoloe/timm) + 4 baseline native
     (yolov8n, yolo26n, yolo11n, yolo12n). Model prioritas duluan: baseline, keluarga
     SGSA/SSGA, sota-fusion, lalu sisanya.
   - Protokol per run: epochs=100, imgsz=640, batch=16, patience=30, seed=0, device=0,
     workers=8, plots=False.
   - Ledger `judol_sweep/ledger.json`: key `model|dataset|fold` -> metrics/error. Restart
     harus skip run yang sudah `status: ok` (resumable).
   - SEMUA exception ditangkap: catat `status: error` + traceback ke ledger, lanjut run
     berikutnya — loop tidak boleh mati karena satu model gagal.
   - Setelah tiap run: update `judol_sweep/summary.json` (mean±std mAP50 & mAP50-95 per
     model per dataset lintas fold), `gc.collect()` + `torch.cuda.empty_cache()`.

## Monitor & perbaikan otonom (loop)
4. Pasang monitor yang membaca ledger tiap 5 MENIT: laporkan run selesai baru + error baru.
   Setelah stabil (≥1 jam tanpa error baru), naikkan interval ke 20–30 menit.
5. Setiap ada `status: error` di ledger:
   a. Baca traceback-nya.
   b. Kalau bug yaml (index Concat salah, arg format salah) → perbaiki yaml-nya.
   c. Kalau bug modul (shape mismatch dsb.) → perbaiki di `ultralytics/nn/modules/block.py`,
      verifikasi dengan build+forward-pass sebelum lanjut.
   d. Kalau OOM → turunkan batch untuk model itu (tambahkan penanganan per-model di skrip).
   e. Hapus entry error dari ledger supaya di-retry, dan kalau prosesnya sudah mati,
      relaunch `run_sweep.py` (resumable, aman).
   f. Ulangi sampai `SWEEP COMPLETE` muncul di log dan ledger bersih dari error.
6. Di akhir: buat rekap hasil (tabel mean±std per model per dataset, ranking vs baseline
   per keluarga) dan laporkan model mana yang melampaui baseline-nya.

## Estimasi & catatan
- Total ±162 model × 3 dataset × 5 fold = ±2.430 run. Di RTX A4000: v2-1 ≈ 8–12 mnt/run,
  v2-9 ≈ 15–25 mnt, judi3k ≈ 45–75 mnt → berminggu-minggu untuk antrian penuh. Model
  prioritas (12 pertama = 180 run) selesai dalam ±4–5 hari; hasilnya bisa dianalisis
  duluan tanpa menunggu antrian penuh.
- Jangan commit apa pun tanpa diminta. Jangan matikan run yang sedang jalan saat
  memperbaiki error — perbaikan berlaku untuk run berikutnya.

---
