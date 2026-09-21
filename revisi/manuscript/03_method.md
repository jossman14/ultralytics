## 2. METHOD

### 2.1. Datasets

Two datasets support two different questions. The first asks whether a frame
contains gambling promotion at all. The second asks which platform is being
promoted.

The **Instagram Online Gambling Dataset** holds 3,000 frames sampled from 300
public Instagram Reels, ten frames per Reel, annotated in two classes, `gambling`
and `nongambling`, at 1,500 instances each [50]. Every annotation is a single
full-frame box written as `0.5 0.5 1.0 1.0`, so although the labels are stored in
detection format the task they define is whole-image classification. We train it
with a detector regardless, because doing so keeps the architecture, schedule, and
evaluation code identical to the multi-class experiment, but we read its results as
classification results throughout Section 3. The frame filenames preserve the
Instagram shortcode of the source Reel, which is what makes the grouped protocol of
Section 2.3 possible.

Figure 1 shows representative frames from the binary dataset and Figure 2 shows
annotated marks from the multi-class dataset.

[[FIGURE:image3.png|Figure 1. Representative frames from the Instagram Online Gambling Dataset]]

[[FIGURE:image4.png|Figure 2. Annotated brand marks from the Multi-Class Gambling Platform Dataset]]

The **Multi-Class Gambling Platform Dataset** holds 443 images with 932 annotated
logo instances across five brands. Table 1 lists the instance count and the
relative size of each brand's mark, measured as the fraction of image area the
bounding box covers. The two columns tell different stories and the size column
matters more for the results in Section 3.4: BK8 has the largest number of
instances and the smallest marks, with a median box covering 0.76% of the image
against 8.67% for Starlight-Princess, a factor of eleven.

**Table 1. Multi-class dataset composition and mark size**

| Class | Instances | Median box area (% of image) | Mean box area (%) | 10th percentile (%) |
|---|---|---|---|---|
| BK8 | 203 | 0.76 | 2.81 | 0.07 |
| Gate-of-Olympus | 171 | 6.26 | 8.19 | 1.16 |
| Princess | 179 | 1.71 | 2.80 | 0.32 |
| Starlight-Princess | 150 | 8.67 | 12.84 | 0.88 |
| Zeus | 229 | 3.69 | 5.99 | 0.79 |

### 2.2. Validation protocols

We evaluate under two five-fold protocols that differ in one respect only, which
is how frames are assigned to folds. Everything else, including fold sizes, model,
hyperparameters, seed, and metric computation, is held constant so that the
difference between the two results is attributable to the split.

The **random protocol** shuffles all 3,000 frames and partitions them with
`KFold(n_splits=5, shuffle=True)`, giving 2,400 training and 600 validation frames
per fold. This is the protocol used by the applied studies we build on, and it is
the protocol our own earlier analysis used.

The **video-disjoint protocol** partitions the 300 source Reels rather than the
frames, using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
with the Reel shortcode as the group key and the frame label as the stratification
target. Fold sizes are unchanged at 2,400 and 600 frames, 240 and 60 Reels. We
assert zero group intersection between each training and validation pair before
training begins.

Figure 3 summarizes the two protocols and the evaluation pipeline.

[[FIGURE:fig_workflow.png|Figure 3. Experimental workflow: dataset construction, the two fold-assignment protocols, detector training, and the detection and explanation evaluation paths]]

The multi-class dataset contains still images rather than video frames and carries
no group structure, so it is evaluated under the random protocol only, using the
five folds published with the dataset.

### 2.3. Detectors and baselines

Reviewers of applied detection work reasonably ask whether a reported result
depends on the specific architecture or would hold for any recent detector. We
therefore train four YOLO generations rather than one: YOLOv8s (11.17M
parameters), YOLOv10s (8.13M), YOLO11s (9.46M), and YOLOv12s (9.28M). Exact
counts vary slightly with the number of output classes and are reported per model
in Table 9.
All four are the small variant of their generation, so the comparison holds
capacity roughly constant.

YOLOv12s is our primary model. Its A2C2f blocks replace the convolutional stages
of earlier generations with area attention, which computes attention within
horizontal or vertical strips of the feature map rather than globally [26]. For
watermark detection the relevant property is that a small rigid mark is
discriminated by its local texture and its relation to nearby image structure, not
by long-range context, and strip attention preserves the former at a cost that
scales linearly rather than quadratically in the number of tokens.

All four models are built from their architecture definitions and trained from
random initialization rather than fine-tuned from COCO weights. This follows the
protocol of the original study, and it keeps the comparison clean: COCO
pretraining transfers unevenly across these four architectures, so a fine-tuned
comparison would measure the quality of each generation's released checkpoint as
much as the architecture itself.

### 2.4. Training configuration

Table 2 lists the hyperparameters, which are identical across all four models,
both datasets, and both protocols. The settings were not tuned per model, by
design: tuning one model and not the others would confound the architecture
comparison in Section 3.3.

**Table 2. Training hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 100 | Training loss plateaus well before this on all folds |
| Early stopping patience | 30 | Stops folds that converge early without truncating slower ones |
| Batch size | 16 | Largest batch fitting two concurrent jobs per 16 GB GPU |
| Image size | 480 x 480 | Preserves the small marks of Table 1; 640 raised cost without raising mAP in pilot runs |
| Optimizer | AdamW | Converged in fewer epochs than SGD on the small multi-class set |
| Initial learning rate | 0.01 | Ultralytics default for AdamW fine-tuning |
| Final LR factor | 0.01, cosine schedule | Anneals to 1e-4, avoiding late-epoch oscillation on 443-image folds |
| Mosaic disabled for final | 10 epochs | Removes composite-image artifacts before the loss settles |
| Mixed precision | Enabled | Halves step time with no measured mAP change |
| Random seed | 0, deterministic | Makes the four-model comparison reproducible |

Table 3 lists the environment.

**Table 3. Experimental environment**

| Component | Specification |
|---|---|
| GPU | 2 x NVIDIA RTX A4000, 16 GB each |
| CPU / RAM | 8 cores / 31.34 GB |
| Python / PyTorch | 3.10.16 / 2.5.1 with CUDA 12.4 |
| Framework | Ultralytics YOLO |

### 2.5. Evaluation thresholds

Architecture comparisons are tested with a paired t-test over the five folds, pairing each baseline against YOLOv12s on the same fold, because the folds are shared across models and a paired test uses that pairing instead of discarding it. Reported p values are uncorrected; the text states the Bonferroni-corrected value where it changes the reading.

Validation uses a non-maximum suppression IoU threshold of 0.7, a maximum of
300 detections per image, and a confidence threshold of 0.001. The confidence
threshold is low enough that the precision-recall curve is traced over
effectively its full range and the average precision integral is not truncated. We report mAP@50, the mean average precision at an IoU
of 0.50, and mAP@50-95, the mean over IoU thresholds from 0.50 to 0.95 in steps of
0.05. Precision, recall, and F1 are reported at the confidence maximizing F1. All
figures are the mean and standard deviation over the five folds.

### 2.6. Multi-scale activation mapping

The explanation module produces a class activation map from the detector's feature
pyramid. Forward hooks capture activations at the three neck layers feeding the
P3, P4, and P5 detection heads, at model indices 14, 17, and 20. For a predicted
box, channel weights are the ReLU of each channel's mean activation inside the box
region, normalized to sum to one, and the map is the weighted channel sum followed
by ReLU. Maps from the three scales are resized to the input resolution and fused
with weights 0.25, 0.35, and 0.40 for P3, P4, and P5, favoring the coarser levels
whose receptive fields match the mark sizes in Table 1.

This procedure uses activations only and computes no gradients, so it is an
activation-based CAM rather than Grad-CAM, and our earlier description of it as
Grad-CAM was imprecise. The distinction matters for how the maps may be
interpreted, and Section 2.7 addresses it.

### 2.7. Measuring explanation quality

Deriving channel weights from inside a predicted box and then scoring the
resulting map against the ground-truth box is partly circular, because the box
conditions the map before the map is evaluated. Our evaluation therefore includes
a prediction-independent variant and two controls.

The **box-conditioned** map is the procedure of Section 2.6 as deployed. The
**global** map computes channel weights over the entire feature map with no box
conditioning, so nothing about the annotation enters the map before it is scored.

We report two scores for each. The **pointing game** counts an explanation as
correct when the map's maximum falls inside the ground-truth box [43]. The
**energy-based pointing game** reports the fraction of total map energy falling
inside the box, which penalizes maps that peak correctly while spreading
elsewhere.

Two controls establish what the scores mean. The **random-box control** scores
each map against a box of the same dimensions placed uniformly at random, giving
the score obtainable from the marginal spatial distribution of the maps alone. The
**model-randomization sanity check** reinitializes all model weights from a normal
distribution and repeats the measurement, following Adebayo et al. [52]: a
saliency method whose score survives weight destruction is reporting image
statistics, not model behavior.

### 2.8. Separating detection cost from explanation cost

A platform may run detection on every upload and explanation only on flagged
items, so a single end-to-end latency figure obscures the deployment decision. We
time the detector forward pass and the CAM extraction separately, with CUDA
synchronization around each, and report them as distinct quantities.
