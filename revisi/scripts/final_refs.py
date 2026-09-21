"""Assemble the revised reference list in citation order from the Crossref-verified pool."""
import json
from pathlib import Path

ROOT = Path("/home/ftib/ultralytics/revisi")
pool = {}
for f in ("refs_resolved.json",):
    pass
# refs_resolved.json is overwritten per batch, so reload the three saved batches
import subprocess
batches = ["refs_spec.json", "refs_spec2.json", "refs_spec3.json"]
for b in batches:
    subprocess.run(["python3", str(ROOT / "scripts/build_refs.py"), str(ROOT / "out" / b)],
                   capture_output=True)
    for x in json.load(open(ROOT / "out/refs_resolved.json")):
        pool[x["key"]] = x

DROP = {"kaggle_placeholder", "logo_survey_mta", "deepfake_moderation", "cam_eval_metrics",
        "adamw", "data_leakage_cv", "dedup_benchmarks", "content_moderation_dl",
        "vlm_moderation", "logo_det_survey2", "attention_mech_det", "xai_object_detection",
        "faithfulness_xai", "ciou", "yolov8_review", "procedia_indo"}

ORDER = [
 "ammar_youtube","muzakir_icisit","muzakir_ijsse","maldini_covert","indo_gambling_web",
 "gambling_sites_ml","trustcom_graph","crossmodal_hidden","cardoso_safer","marionneau_duty",
 "murch_temporal","yolo_logo_icacc","logo_yolo_ice3is","logo_yolov8_icdabi","logo_cyclegan",
 "su_scalable_logo","open_logo_cviu","vehicle_logo_cee","multiscale_logo_displays",
 "brand_packaging","dr_yolo_livestream","deep_logo_survey","sapkota_yolo_review","yolov10",
 "rtdetr","YOLOV12_PREPRINT","yolov12_exam","ijece_yolo","layn_yolov8","swin_yolov5_sensors",
 "ghost_yolov8","lkd_yolov8","setyanto_kd","lms_yolo","small_object_survey","attention_survey",
 "diou","dfl_gfl","gradcam_journal","gradcampp","eigencam","scorecam","excitation_bp",
 "explaining_yolo","sanity_checks","xai_review_ieee","xai_survey_2023","leakage_crisis",
 "phash_dup","DATASET_KAGGLE","DATASET_ROBOFLOW",
]

MANUAL = {
 # Verified against DataCite (arXiv). The only preprint retained in the list.
 "YOLOV12_PREPRINT": 'Y. Tian, Q. Ye, and D. Doermann, "YOLOv12: Attention-Centric Real-Time '
                     'Object Detectors," arXiv:2502.12524, 2025, doi: 10.48550/arXiv.2502.12524.',
 # Proceedings DOI is registered outside Crossref (TU Graz / OAGM).
 "explaining_yolo": 'A. Kirchknopf, D. Slijepcevic, I. Wunderlich, M. Breiter, J. Traxler, and '
                    'M. Zeppelzauer, "Explaining YOLO: Leveraging Grad-CAM to Explain Object '
                    'Detections," in Proc. Joint Austrian Computer Vision and Robotics Workshop '
                    '(ACVRW), 2022, doi: 10.3217/978-3-85125-869-1-13.',
 "DATASET_KAGGLE": 'A. M. Faisal, "Online Gambling Logo in Instagram Reels Video," Kaggle, 2025. '
                   '[Online]. Available: https://www.kaggle.com/datasets/zielisme/'
                   'online-gambling-logo-in-instagram-reels-video',
 "DATASET_ROBOFLOW": 'B. Farrel, "Judol Detection v2 Dataset," Roboflow Universe, 2025. [Online]. '
                     'Available: https://universe.roboflow.com/binus-farrel/judol-detection-v2',
}

lines, missing, preprints = [], [], []
for i, k in enumerate(ORDER, 1):
    if k in DROP:
        continue
    if k in MANUAL:
        txt = MANUAL[k]
    elif k in pool:
        txt = pool[k]["ieee"].replace("*", "")
        if pool[k]["type"] in ("posted-content",):
            preprints.append(k)
    else:
        missing.append(k); continue
    lines.append(f"[{len(lines)+1}]\t{txt}")

out = "\n".join(lines)
(ROOT / "out/references_ieee.md").write_text(out + "\n")
print(out)
print(f"\nTOTAL: {len(lines)} references")
print("preprints kept:", ["YOLOV12_PREPRINT"] + preprints)
print("missing from pool:", missing)
