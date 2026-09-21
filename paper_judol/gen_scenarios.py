"""Generate scenario ablation configs from the proposed yolov8-sgsa.yaml base.

3 scenarios, each an ablation axis on the proposed SGSA-YOLO:
  A) head attention   : which attention at P3/P4/P5 detection scales
  B) backbone block   : which lightweight conv block in the backbone
  C) neck fusion      : extra module in the neck to recover mAP50-95

Each variant is a small module-swap on the base; written to cfg/models/v8/.
"""
from pathlib import Path

CFG = Path("/home/ftib/ultralytics/ultralytics/cfg/models/v8")
BASE = (CFG / "yolov8-sgsa.yaml").read_text()


def w(name, text):
    (CFG / name).write_text(text)
    print("wrote", name)


# ---- Scenario A: head attention swap (replace SSGA at lines 22/23/24) ----
def head_variant(mod):
    t = BASE
    t = t.replace("[15, 1, SSGA, []] # 22 SSGA on P3", f"[15, 1, {mod}, []] # 22 {mod} on P3")
    t = t.replace("[18, 1, SSGA, []] # 23 SSGA on P4", f"[18, 1, {mod}, []] # 23 {mod} on P4")
    t = t.replace("[21, 1, SSGA, []] # 24 SSGA on P5", f"[21, 1, {mod}, []] # 24 {mod} on P5")
    return t


# A0: no attention -> Detect directly on 15/18/21
a0 = BASE
for line in ["  - [15, 1, SSGA, []] # 22 SSGA on P3\n",
             "  - [18, 1, SSGA, []] # 23 SSGA on P4\n",
             "  - [21, 1, SSGA, []] # 24 SSGA on P5\n"]:
    a0 = a0.replace(line, "")
a0 = a0.replace("[[22, 23, 24], 1, Detect, [nc]] # Detect(P3, P4, P5)",
                "[[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5) - no attention")
w("yolov8-sgsaA0-noattn.yaml", a0)
w("yolov8-sgsaA1-cbam.yaml", head_variant("CBAM"))
w("yolov8-sgsaA2-eca.yaml", head_variant("ECA"))
w("yolov8-sgsaA3-gam.yaml", head_variant("GAM_Attention"))

# ---- Scenario B: backbone block swap (C2fStar -> other blocks) ----
for name, mod in [("B1-ghost", "C2fGhost"), ("B2-inception", "C2fInception"),
                  ("B3-dense", "C2fDense"), ("B4-c2f", "C2f")]:
    w(f"yolov8-sgsa{name}.yaml", BASE.replace("C2fStar", mod))

# ---- Scenario C: neck fusion enhancement ----
# C1: add CBAM after each neck C2fGhost output (P3/P4/P5 pre-attention)
# Simplest structural swap: replace neck C2fGhost with C2fInception/CFCGLU-flavoured fusion.
# C1 neck-cbam: insert CBAM by swapping the 3 neck C2fGhost to C2fGhost + rely on head; instead
#   we make neck heavier via C3Ghost (reparam-friendly) — a real, buildable fusion change.
w("yolov8-sgsaC1-neckdense.yaml", BASE.replace("[-1, 3, C2fGhost,", "[-1, 3, C2fDense,"))
w("yolov8-sgsaC2-neckinception.yaml", BASE.replace("[-1, 3, C2fGhost,", "[-1, 3, C2fInception,"))
w("yolov8-sgsaC3-neckstar.yaml", BASE.replace("[-1, 3, C2fGhost,", "[-1, 3, C2fStar,"))

print("DONE")
