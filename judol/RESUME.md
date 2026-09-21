# Melanjutkan Sweep Judol

Sweep di-pause pada **386/1620 run selesai (24%), 0 error**.

Semua state tersimpan di `judol_sweep/ledger.json` (386 entry `ok`). Sweep **resumable**:
run yang sudah `ok` otomatis di-skip, jadi tinggal luncurkan ulang worker.

## Cara lanjut

```bash
cd /home/ftib/ultralytics
bash judol/launch_parallel.sh 2      # 2 worker paralel, resume dari ledger
```

Lalu pantau:
```bash
tail -f judol_sweep/worker_0.log judol_sweep/worker_1.log
# atau cek ringkas:
python -c "import json; d=json.load(open('judol_sweep/ledger.json')); print(sum(1 for v in d.values() if v.get('status')=='ok'),'ok /1620')"
```

## Konfigurasi saat pause (sudah tertanam di skrip)
- `workers-per-job = 2` (di `judol/launch_parallel.sh`) — dikurangi dari 4 untuk mencegah swap-thrashing.
- `cache = "disk"` untuk judi3k, `"ram"` untuk v2-1 (di `judol/run_sweep.py`) — cegah RAM habis.
- Protokol: 100 epoch, imgsz 640, batch 16, patience 30, seed 0, GPU 0.
- Dataset: v2-1 + judi3k (v2-9 sengaja di-drop, augmentasi berat).

## Hasil sementara (v2-1, medan uji utama)
- Terbaik absolut: **yolov8-spd-shuffle-simam 0.8836**
- Delta terbesar: **yolo26-shuffle-spd-eca +0.0351** (signifikan)
- 10 model melampaui baseline keluarganya — SEMUA berbasis **ShuffleNet + SPDConv + attention ringan**.

Rekap penuh: `python -c "import json; [print(k,v.get('v2-1',{}).get('mAP50_mean')) for k,v in json.load(open('judol_sweep/summary.json')).items()]"`

## Pindah ke GPU lebih besar (RunPod)
Salin repo + folder `judol/` (dengan `judol_sweep/ledger.json` untuk lanjut, atau kosongkan untuk mulai baru) + dataset `*_strat5fold/`.
Rekomendasi GPU: A100 PCIe ($1.39/hr, 80GB, 31 vCPU). Naikkan konkurensi: `bash judol/launch_parallel.sh 10` + edit `--workers-per-job 3` di launcher.
