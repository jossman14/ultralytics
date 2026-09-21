# Handoff: revisi mayor IJAAS Paper ID 22817

Terakhir diperbarui: 2026-09-21. **Status: revisi selesai.** Dokumen ini merekam seluruh progres revisi supaya
sesi berikutnya bisa melanjutkan tanpa membaca ulang transkrip.

## 1. Konteks

Paper "Explainable AI for Covert Online Gambling Detection Using YOLOv12 and Grad-CAM"
(Tanzilal Mustaqim, Pima Hani Safitri, Yasinta Romadhona) mendapat keputusan
**major revisions required** dari IJAAS. Naskah asli ada di
`/home/ftib/ultralytics/[1] paper judol ijaas.docx` (read-only, jangan diubah).
Batas waktu resubmit 8 minggu, format MS Word, ID sama, tab "Review", sebagai
"Author Version".

Instruksi pengguna: kerjakan sampai semua catatan reviewer selesai, boleh training
ulang, gunakan semua GPU, dan terapkan skill antislop.

## 2. Temuan utama yang mengubah arah paper

### 2.1 Dataset binary bukan dataset deteksi

Seluruh 3.000 anotasi di `judol/dataset_judi_online_yolo_5fold` tertulis persis
`0.5 0.5 1.0 1.0`: satu kotak per gambar, menutup seluruh frame.

```
binary             n=3000  median_area=100.000%  full-image boxes=3000 (100.0%)
binary_videofold   n=3000  median_area=100.000%  full-image boxes=3000 (100.0%)
multiclass         n= 932  median_area=  3.095%  full-image boxes=2   (0.2%)
```

Ketahuan karena semua skor XAI pada dataset binary keluar tepat 1.0, termasuk
kontrol kotak acak dan sanity check model teracak, yang mustahil kecuali kotak
ground truth menutup gambar.

Konsekuensi:
1. mAP@50-95 binary mengukur klasifikasi, bukan lokalisasi.
2. Klaim "Grad-CAM menemukan morphological markers" tidak bisa diuji di data itu.
3. Ini jawaban urutan pertama untuk Reviewer C soal performa binary yang luar biasa
   tinggi, mendahului isu leakage.

Akibatnya judul dan framing diubah menjadi dua tugas terpisah: klasifikasi
whole-image dan lokalisasi logo.

### 2.2 Leakage terkuantifikasi

100% frame validasi punya frame saudara dari Reel yang sama di training pada
protokol random. Perbandingan 5-fold YOLOv12s:

| Protokol | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Random 5-fold | 98.96 +/- 0.41 | 99.51 +/- 0.21 | 99.23 +/- 0.16 | 99.47 +/- 0.04 | 99.45 +/- 0.07 |
| Video-disjoint | 96.59 +/- 1.69 | 96.63 +/- 1.36 | 96.60 +/- 1.39 | 99.04 +/- 0.31 | 98.47 +/- 1.11 |

Efeknya kecil pada rata-rata (mAP@50 turun 0.43 poin) tetapi besar pada
ketidakpastian: standar deviasi metrik ketat melebar sekitar sepuluh kali lipat
(0.07 ke 1.11).

### 2.3 Penyebab kegagalan per kelas

5 fold YOLOv12s multiclass:

```
BK8                  62.99 +/- 11.42 mAP50 | 39.54 mAP50-95 | recall 50.30 +/- 12.75
Gate-of-olympus      90.54 +/-  5.96       | 54.37          | 88.58
Zeus                 91.58 +/-  4.69       | 60.39          | 88.44
Princess             93.17 +/-  2.62       | 62.60          | 89.47
Starlight-Princess   96.81 +/-  2.30       | 71.78          | 95.04
```

Penyebabnya ukuran mark (BK8 median 0.76% luas frame vs Starlight-Princess 8.67%,
selisih 11 kali), bukan jumlah instance (BK8 justru terbanyak, 203). Kontras tidak
diukur, jadi jangan diklaim.

### 2.4 Hasil XAI yang valid (multiclass saja, 5 fold)

```
pg_box    47.7 +/- 7.4     eb_glob   11.0 +/- 2.0
pg_glob   39.0 +/- 14.0    eb_rand   10.5 +/- 1.5
pg_rand   11.7 +/- 3.2     area      10.0 +/- 1.5
pg_sanity  7.3 +/- 10.0    eb_sanity == area (persis, peta seragam)
```

Bacaan jujurnya: puncak peta mendarat di tempat yang benar jauh lebih sering
daripada kebetulan, tetapi massa peta nyaris tidak lebih terkonsentrasi di mark
dibanding wilayah acak seluas itu. Ini kesesuaian lokalisasi, bukan causal
faithfulness.

### 2.5 Reproduksi naskah asli terkonfirmasi

Binary fold 1 mAP@50 = 0.9949, sama persis dengan 99.49% yang disubmit.
Multiclass 87.02 +/- 2.50 vs 87.14% yang disubmit. mAP@50-95 57.74 +/- 1.50 vs
58.76%.

## 3. Status sweep training

4 model x 3 dataset x 5 fold = 60 run. Hyperparameter, fold, dan seed identik
supaya perbandingan mengisolasi arsitektur.

Parameter: YOLOv8s 11.126.358 | YOLOv10s 7.218.774 | YOLO11s 9.413.574 |
YOLOv12s 9.231.654.

Sweep **selesai 60/60**. Hasil final:

| Dataset | Model | P | R | mAP@50 | mAP@50-95 | Inferensi |
|---|---|---|---|---|---|---|
| binary | YOLOv8s | 99.45 | 99.50 | 99.50 +/- 0.00 | 99.49 | 2.10 ms |
| binary | YOLOv10s | 98.32 | 97.80 | 99.36 +/- 0.18 | 98.78 | 2.55 ms |
| binary | YOLO11s | 98.90 | 99.02 | 99.49 +/- 0.01 | 99.29 | 2.09 ms |
| binary | YOLOv12s | 98.96 | 99.51 | 99.47 +/- 0.04 | 99.45 | 2.75 ms |
| videofold | YOLOv8s | 97.94 | 97.29 | 99.37 +/- 0.04 | 99.31 | 2.21 ms |
| videofold | YOLOv10s | 96.23 | 95.45 | 98.76 +/- 0.54 | 98.02 | 2.41 ms |
| videofold | YOLO11s | 96.40 | 97.01 | 99.14 +/- 0.47 | 98.77 | 1.91 ms |
| videofold | YOLOv12s | 96.59 | 96.63 | 99.04 +/- 0.31 | 98.47 | 2.96 ms |
| multiclass | YOLOv8s | 94.67 | 83.25 | 88.55 +/- 4.07 | 60.15 | 3.08 ms |
| multiclass | YOLOv10s | 85.72 | 77.43 | 85.04 +/- 2.39 | 56.89 | 3.68 ms |
| multiclass | YOLO11s | 93.81 | 82.46 | 87.89 +/- 3.60 | 58.84 | 5.55 ms |
| multiclass | YOLOv12s | 91.91 | 82.37 | 87.02 +/- 2.50 | 57.74 | 7.42 ms |

**Temuan: YOLOv12s bukan yang terbaik.** Peringkat tiga dari empat pada tugas
multiclass. Uji t berpasangan antar fold, tiap baseline dipasangkan dengan YOLOv12s
pada fold yang sama:

- Binary kedua protokol: tidak ada yang berbeda signifikan, semua p > 0.06.
- Multiclass, YOLOv8s unggul 2.42 poin mAP@50-95, p = 0.017, dan unggul di kelima
  fold tanpa kecuali.
- Multiclass, YOLOv10s tertinggal 1.98 poin mAP@50, p = 0.008.
- YOLO11s tidak berbeda signifikan dari YOLOv12s pada metrik mana pun.

Tiga perbandingan per metrik, jadi setelah koreksi Bonferroni keunggulan YOLOv8s
berada di ambang (0.051) sementara defisit YOLOv10s bertahan (0.024).

Sisi biaya memperlebar jarak: YOLOv12s butuh 7.42 ms per gambar di multiclass
melawan 3.08 ms YOLOv8s, jadi 2.4 kali lebih lambat sambil membawa parameter 17%
lebih sedikit. Praktisi yang memilih detektor berdasarkan bukti studi ini akan
memilih YOLOv8s.

Paragraf 3.3, kesimpulan bagian "On architecture", dan catatan uji berpasangan di
metode 2.5 sudah ditulis sesuai temuan ini. YOLOv12s tetap dilaporkan sebagai model
utama karena ia adalah sistem yang direvisi, bukan karena ia menang.

## 4. Proses yang sedang berjalan

Tidak ada. Sweep 60/60 selesai, rantai analisis selesai, kedua .docx terbangun.

## 4b. Kelengkapan tabel dan gambar (21 September 2026)

Pemeriksaan ulang `revisi/out/manuscript_ijaas.docx` terhadap naskah asli menemukan
bahwa versi revisi kehilangan sebagian tabel dan seluruh panel Grad-CAM. Penyebabnya
bukan gambarnya hilang: `fig_cam_multiclass.png` dan `fig_cam_binary_videofold.png`
sudah dihasilkan `make_figures.py` tetapi tidak pernah dirujuk oleh satu pun
direktif `[[FIGURE:...]]`. Yang diperbaiki:

- Panel CAM digambar ulang di ruang input 640x640 milik CAM itu sendiri. Sebelumnya
  tiap panel memakai rasio aspek sumbernya, jadi grid tidak rata dan judul kolom
  duduk di ketinggian berbeda. Panel binary juga dideduplikasi per Reel, karena
  empat frame berurutan dari satu Reel adalah near-duplicate.
- Panel yang salah klasifikasi sekarang menyebut kelas sebenarnya di judulnya.
  Panel kedua gambar binary adalah frame berwatermark judi yang diprediksi
  nongambling, dan itu dinyatakan, bukan diganti dengan contoh yang benar.
- Enam tabel baru dibangun dari `runs/*/*/fold_*/metrics.json` dan `results.csv`:
  per-fold binary dua protokol (Tabel 6), per-kelas binary (Tabel 7), biaya dan loss
  per fold binary (Tabel 8), per-fold multiclass (Tabel 10), biaya dan loss per fold
  multiclass (Tabel 12). Generatornya ada di `fill_tokens.py:write_fold_tables`.
- Bagian BIOGRAPHIES OF AUTHORS ditambahkan, `revisi/manuscript/08_biographies.md`.
- `build_docx.py` sekarang memakai gaya bawaan template IJAAS: `Table Title` untuk
  judul tabel, `figure caption` untuk keterangan gambar, `Abstract`, `references`,
  `AutoBiography`, `paper title`. Lebar gambar mengikuti lebar teks dan diperkecil
  bila tingginya melebihi setengah blok teks, menggantikan lebar tetap 340 pt.
- Bug yang ditemukan sambil jalan: seluruh 52 referensi menyatu jadi satu paragraf
  sepanjang 12.000 karakter, karena entri IEEE ditulis satu baris tanpa baris
  kosong pemisah. Sekarang tiap entri jadi paragraf sendiri.

Hasil akhir: 13 tabel bernomor plus tabel CRediT dan biografi, 12 gambar, penomoran
berurutan tanpa lompatan, setiap gambar disebut di teks sebelum kemunculannya
(catatan Reviewer A nomor 5), nol token belum terisi, nol em dash.

## 5. Sisa pekerjaan

Nol placeholder di naskah. Yang tersisa hanya yang tergantung penulis, bertanda
`[AUTHORS TO COMPLETE]` atau `[AUTHORS TO VERIFY]`:

- Alamat jalan dan email corresponding author, `revisi/manuscript/01_front.md`
- Acknowledgments, funding statement, verifikasi CRediT, `06_backmatter.md`
- Ketentuan berbagi data Instagram di Data Availability
- Foto dan biografi tiap penulis, `08_biographies.md`

Setelah itu naskah siap disubmit.

## 6. Tanggapan terhadap tiap catatan reviewer

### Editor
Minimal 25 referensi, mayoritas jurnal: selesai, 52 referensi.

### Reviewer A
1. Abstrak: ditulis ulang, 249 kata, framing dua tugas. Selesai.
2. Gap dan kontribusi eksplisit di pendahuluan: selesai, empat kontribusi bernomor.
3. Baseline comparison: selesai, empat generasi YOLO, Tabel 6 dan Bagian 3.3,
   dengan uji t berpasangan antar fold.
4. Diskusi limitasi termasuk BK8 dan metrik lokalisasi ketat: selesai di 3.4 dan 3.5.
5. Gambar setelah disebut: selesai, diatur `build_docx.py`.
6. IEEE style dengan DOI: selesai, 50 dari 52 ber-DOI, 2 sisanya URL dataset.
7. Minimal 25 referensi: selesai, 52.
8. Sitasi menggantung [17] dan [20]: selesai, nol menggantung, nol tak tersitasi.
9. Preprint maksimal 2: selesai, tepat 2 (YOLOv12 dan Adebayo).

### Reviewer B
Ukuran dataset di abstrak, pemisahan binary dan multiclass, mAP@50-95 multiclass
yang jauh lebih rendah dilaporkan, klaim morphological markers dilunakkan dan
diganti angka XAI kuantitatif, mean +/- SD konsisten, threshold dilaporkan
(NMS IoU 0.7, max 300 deteksi, conf 0.001), "mAP@50-90" dikoreksi jadi "mAP@50-95",
hasil per kelas ditonjolkan, latensi deteksi dipisah dari biaya pipeline penuh.
Semua selesai.

### Reviewer C
Interpretasi kritis generalisasi, limitasi, reliabilitas XAI, deployment praktis;
perbandingan dengan studi terbaru; pembahasan kritis performa binary yang sangat
tinggi termasuk kemungkinan korelasi sumber; pengakuan heatmap bersifat kualitatif;
limitasi domain shift. Kesimpulan memisahkan binary dan multiclass, tidak menyebut
sistem "robust", menjelaskan Grad-CAM sebagai bukti kualitatif, menambah future
work. Semua selesai.

## 7. Berkas

| Berkas | Isi |
|---|---|
| `revisi/out/manuscript_ijaas.docx` | Naskah revisi, keluaran final |
| `revisi/out/response_to_reviewers.docx` | Surat balasan, keluaran final |
| `revisi/response_to_reviewers.md` | Sumber surat balasan |
| `revisi/scripts/build_letter.py` | Perender surat, memakai tokens_filled.json yang sama dengan naskah |
| `revisi/manuscript/*.md` | Sumber naskah, 8 bagian |
| `revisi/scripts/` | Training, XAI, agregasi, pembangun docx |
| `revisi/runs/` | Metrik per fold |
| `revisi/out/xai_eval.json` | Skor XAI, hanya multiclass |
| `revisi/figures/` | Gambar hasil dan gambar dari naskah asli |
| `[1] paper judol ijaas.docx` | Naskah asli, jangan diubah |

## 8. Catatan operasional

- Hook shell `rtk` merusak keluaran `grep` dan `ps | grep`. Pakai `pgrep` atau
  glob Python.
- `xai_eval.py` sengaja dibatasi ke multiclass. Baris binary yang degenerate sudah
  dihapus dari `xai_eval.json`, bukan difilter, karena `fill_tokens.py` merata-rata
  seluruh dataset dan sempat membuat abstrak melaporkan kontrol acak 70.6%
  sementara isi badan menulis 11.7%.
- Scanner antislop ada di scratchpad, `slop.py`. Hasil stabil 4 hit, semuanya sah:
  satu judul referensi, dua kutipan Reviewer C di surat balasan, dan satu kalimat
  kesimpulan yang memang menolak kata tersebut.
- **Bug yang sudah diperbaiki di `finalize.sh`:** skrip menunggu dengan
  `pgrep -f "train_baselines.py --slot"`. Shell peluncurnya membawa seluruh teks
  skrip di cmdline-nya sendiri, termasuk string itu, sehingga pgrep mencocokkan
  induknya sendiri dan loop tidak pernah keluar. Training sebenarnya sudah habis
  berjam-jam sebelumnya dan kedua GPU menganggur di 0%. Penantiannya dihapus, bukan
  diberi guard, karena training memang sudah selesai. Pelajaran umum: jangan
  pgrep pola yang muncul di cmdline proses yang menjalankan pgrep itu.
- Aturan repo: jangan push ke `main`, jangan force push, kerja di worktree dan
  branch terpisah bila membuat PR.
- Email pengguna hanya dipakai sebagai kontak User-Agent untuk Crossref dan
  DataCite.

## 9. Perintah untuk melanjutkan

Sweep sudah 60/60 dan rantai sudah dijalankan. Jalankan ulang hanya bila sumber
markdown diubah:

```bash
cd /home/ftib/ultralytics
PYTHONPATH=. python3 revisi/scripts/aggregate.py
PYTHONPATH=. python3 revisi/scripts/fill_tokens.py
PYTHONPATH=. python3 revisi/scripts/make_figures.py
PYTHONPATH=. python3 revisi/scripts/build_docx.py    # 0 unresolved placeholders
PYTHONPATH=. python3 revisi/scripts/build_letter.py
```

Cek progres sweep:

```bash
python3 -c "import glob; print(len(glob.glob('revisi/runs/*/*/fold_*/metrics.json')),'/60')"
```

## 4c. Template dan sitasi klik (21 September 2026)

`revisi/scripts/build_docx.py` ditulis ulang agar formatnya mengikuti versi yang
disubmit (`[1] paper judol ijaas.docx`), bukan gaya style bawaan template yang
dipakai sebelumnya:

- Halaman A4 8.27 x 11.69 in, margin kiri 1.18, kanan/atas/bawah 0.98, satu kolom.
- Judul `Title` 16pt rata tengah, penulis `Normal` tebal rata tengah, afiliasi 8pt.
- Kotak article info 5x3 (Article Info / Abstract / Keywords / lisensi CC /
  Corresponding Author) dengan garis ganda atas-bawah, menggantikan heading
  "ABSTRACT" polos.
- Badan teks `Normal` rata kiri-kanan dengan indent baris pertama 0.5 in.
- Judul bagian bernomor pakai Heading, back matter (ACKNOWLEDGMENTS sampai
  BIOGRAPHIES) pakai `Normal` tebal seperti aslinya.
- Caption tabel rata tengah di atas tabel, caption gambar rata tengah di bawah
  gambar, gambar rata tengah.
- Tabel data memakai tiga garis (atas header, bawah header, bawah baris terakhir);
  tabel CRediT dan biografi tetap full grid.
- Referensi 8pt rata kiri-kanan dengan hanging indent 0.39 in.

Sitasi `[n]` di badan teks sekarang berupa hyperlink internal ke bookmark `ref_n`
pada entri referensi. Verifikasi build terakhir: 52 bookmark, 52 target tertaut,
0 tautan menggantung, 0 token kosong, 0 em dash, 249 paragraf, 16 tabel, 12 gambar,
8.829 kata.

Catatan: di mesin ini tidak ada LibreOffice/Pandoc, jadi pemeriksaan visual PDF
belum bisa dilakukan; verifikasi dilakukan pada struktur dan properti paragraf docx.
