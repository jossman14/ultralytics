### 1.1. Related work

**Gambling promotion detection.** Most published systems read text. Ammar et al.
classify Indonesian YouTube comments [1], and Muzakir et al. combine AutoML with
convolutional classifiers over post text [2], [3]. The multimodal systems are
closer to our setting: Maldini et al. pair Faster R-CNN with Tr-OCR to recover
text rendered into images [4], and Ajhari et al. use cross-modal attention for
promotion hidden across modalities in multilingual posts [8]. Liem et al. and Min
and Lee detect promotional websites from page-level features [5], [6]. A parallel
line in gambling-harm research builds models over behavioural rather than content
data [9], [10], and Louderback et al. raise a point that bears directly on ours,
namely that such models lose accuracy as the behaviour they were fitted to drifts,
so temporal stability must be measured and not assumed [11]. Our finding is the
spatial analogue of theirs: a model fitted to 300 Reels does not transfer to the
301st.

Perceptual hashing is the established alternative for finding known marks and is
cheaper than detection [49]. It matches near-duplicates of images it has already
seen, which makes it effective against reposts and ineffective against a familiar
logo composited into a new frame at a new scale. Detection and hashing are
complementary, and the case for detection rests on the novel-composition case.

**Logo and trademark detection.** The task has a developed literature. Su et al.
scale logo detection through self co-learning on noisy web data [16] and address
the open-set case where test brands are unseen during training [17], which is the
generalization axis our Section 3.8 identifies as untested for gambling marks.
Domain-specific detectors cover vehicle logos [18], multi-scale marks [19],
packaging [20], and livestreaming overlays [21], the last being the closest setting
to gambling watermarks. Earlier YOLO-based logo detectors [12], [22] establish the
single-stage approach we follow.

**One-stage detection and small objects.** Sapkota et al. review the YOLO series
across a decade [23], and the generations we compare are documented in [24], [25],
[26], with applications of the newest to surveillance [27], [28]. The
small-object problem that dominates our per-class results has its own literature:
Cheng et al. survey benchmarks and methods [35], while LAYN [29], Swin-augmented
YOLOv5 [30], Ghost-YOLO v8 [31], and LMS-YOLO [34] add attention, transformer
backbones, or lightweight multi-scale stages specifically to recover small
targets. Knowledge distillation offers a second route, compressing a
high-resolution teacher into a deployable student [32], [33]. Attention mechanisms
in detection are surveyed by Guo et al. [36], and the localization losses our
models optimize derive from [37], [38].

**Explainability for detection.** Grad-CAM [39] and its successors [40], [41],
[42] produce class-discriminative maps for classifiers, and adapting them to
detectors requires choosing which head and which scale to attribute, as in the
YOLO-specific treatment of [44]. The evaluation side is what applied work tends to
skip. Zhang et al. introduced the pointing game [43], Wang et al. added the
energy-based variant [42], and Adebayo et al. showed that several popular methods
pass visual inspection while failing weight-randomization tests [52], a result
reinforced by later sanity-check studies [45]. Surveys of the field [46], [47]
treat quantitative evaluation as a requirement rather than an option. We adopt that
position.

**What is missing.** Across these lines, no study we found reports gambling logo
detection under a grouped split, and none reports explanation quality against
controls. The present work supplies both.
