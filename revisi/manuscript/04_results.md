## 3. RESULTS AND DISCUSSION

### 3.1. Leakage in the conventional protocol

Before reporting accuracy we report what the conventional protocol measures. In
each of the five random folds, every one of the 600 validation frames has at least
one sibling frame from the same Reel in the training set. The leakage rate is
100.0% in all five folds, and all 300 source Reels appear in the training set of
every fold, with between 260 and 276 of them also appearing in validation (Table
4).

**Table 4. Source-video overlap under the random 5-fold protocol**

| Fold | Train frames | Val frames | Reels in train | Reels in val | Reels in both | Val frames with a training sibling |
|---|---|---|---|---|---|---|
| 0 | 2,400 | 600 | 300 | 276 | 276 | 100.0% |
| 1 | 2,400 | 600 | 300 | 260 | 260 | 100.0% |
| 2 | 2,400 | 600 | 300 | 268 | 268 | 100.0% |
| 3 | 2,400 | 600 | 300 | 270 | 270 | 100.0% |
| 4 | 2,400 | 600 | 300 | 263 | 263 | 100.0% |

This is a property of the dataset construction, not of any model. Ten frames drawn
from one short Reel share a background, a subject, a colour grade, a compression
history, and usually the same logo in the same screen position. A model that has
memorized nine of them can classify the tenth without learning anything
transferable about gambling logos. The conventional protocol therefore cannot
distinguish a detector that recognizes brands from one that recognizes Reels.

### 3.2. What the leakage costs

Table 5 reports YOLOv12s on the binary task under both protocols. The model,
hyperparameters, seed, and fold sizes are identical; only the assignment of frames
to folds differs.

**Table 5. Binary task, YOLOv12s, random versus video-disjoint folds (mean +/- SD over 5 folds)**

| Protocol | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Random 5-fold | {{bin.p}} | {{bin.r}} | {{bin.f1}} | {{bin.map50}} | {{bin.map}} |
| Video-disjoint 5-fold | {{vid.p}} | {{vid.r}} | {{vid.f1}} | {{vid.map50}} | {{vid.map}} |
| Difference | {{d.p}} | {{d.r}} | {{d.f1}} | {{d.map50}} | {{d.map}} |

Under the random protocol {{LEAK_PCT}} of validation frames have a sibling frame
from the same Reel in the training split, so on every fold the model was scored on
Reels it had already trained on. The cost of that is smaller than the completeness
of the leakage suggests. Detection quality holds up: mAP@50 falls by only
{{d.map50}} and mAP@50-95 by {{d.map}} when the folds are made video-disjoint. The
operating point moves more, with precision down {{d.p}}, recall down {{d.r}}, and
F1 down {{d.f1}}, because those three are read at a single confidence threshold and
are more sensitive to frames the model has effectively memorized than the
threshold-free average precision is.

The larger effect is on the uncertainty rather than the mean. Standard deviation
across folds widens from {{bin.map50.sd}} to {{vid.map50.sd}} points on mAP@50 and
from {{bin.map.sd}} to {{vid.map.sd}} points on mAP@50-95, which is roughly an
order of magnitude on the strict metric. Random folds therefore overstated the accuracy slightly and
the consistency considerably, because each fold's validation set was drawn from the
same 300 Reels as its training set, leaving the folds little room to disagree with
one another. We take this as the more useful warning for work on
short-form video: a frame-level split can leave the headline accuracy nearly
intact while erasing the between-fold variation that tells a reader how much to
trust it. Table 6 shows the folds individually, which is where that claim is
checkable: the five random folds agree with each other to within 0.10 points of
mAP@50, while the video-disjoint folds spread from 98.56 to 99.38, and fold 0
loses nearly three points of mAP@50-95 against the best fold.

**Table 6. Binary task per fold, YOLOv12s, under both protocols (%)**

[[INCLUDE:table_folds_binary.md]]

These figures must be read with a property of the binary annotations in mind.
Every label in that dataset is the full frame, encoded as `0.5 0.5 1.0 1.0`, with
exactly one box per image and two classes, gambling and nongambling, at 1500
instances each. The binary task is therefore whole-image classification expressed
in detection format, not object detection. Nothing in it requires the model to
find where a mark is, and a predicted full-frame box scores a near-unity IoU by
construction.

Two consequences follow. First, the binary mAP@50-95 of {{bin.map}} is not evidence
of precise localization; it is the classification decision re-reported through a
metric whose localization component is trivially satisfied. We keep both columns in
Table 5 for continuity with the submitted version, but the honest reading of the
binary rows is image-level accuracy. Second, the near-ceiling score is explained
first by the degeneracy of the task and only second by the leakage: distinguishing
frames that carry a gambling overlay from frames that do not is a far easier problem
than locating the overlay, and both protocols measure the easier problem. Section
3.5 returns to what this does to the explanation analysis, which cannot be scored
against full-frame ground truth at all.

Figure 4 shows the same comparison across all five metrics.

[[FIGURE:fig_protocol.png|Figure 4. Binary task under the random and video-disjoint protocols. Bars are the mean over five folds and error bars are one standard deviation.]]

The practical reading is that the video-disjoint figure, not the random-fold
figure, is what a moderation team should plan capacity and human-review thresholds
around, because a deployed system meets Reels it has never seen. We report both
because the comparison is the finding; reporting only the higher number would
repeat the error we are documenting.

The two binary classes are not equally easy, and the asymmetry matters for
moderation. Table 7 splits the video-disjoint result by class. Recall on the
gambling class is the lower of the two, so the errors that survive are missed
promotions rather than false accusations against ordinary posts, which is the
safer direction for an automated screen but also the direction that lets material
through.

**Table 7. Per-class results on the binary task, video-disjoint folds, YOLOv12s (mean +/- SD over 5 folds)**

[[INCLUDE:table_perclass_binary.md]]

Table 8 reports the per-fold inference cost and the final-epoch losses, and its
box-loss column is a second piece of evidence that the binary task is not a
localization task. Training box loss settles near 0.05 against 1.13 on the
multi-class task in Table 12, because a model that always predicts the whole frame
is already correct about geometry on every image. The classification loss carries
what little the task asks of the model.

**Table 8. Per-fold inference cost and final-epoch losses, binary task, video-disjoint folds**

[[INCLUDE:table_eff_binary.md]]

Figure 5 shows the loss trajectories behind those endpoints.

[[FIGURE:fig_loss_binary_videofold.png|Figure 5. Training and validation loss trajectories on the binary task under the video-disjoint protocol. Shaded bands are one standard deviation across the five folds.]]

### 3.3. Architecture comparison

Table 9 reports all four YOLO generations under identical conditions.

**Table 9. Detector comparison across datasets and protocols (mean +/- SD over 5 folds)**

[[INCLUDE:table_arch.md]]

On the binary task no generation separates from another under either protocol.
Mean mAP@50 spans 99.36% to 99.50% across the four models under random folds and
98.76% to 99.37% under video-disjoint folds, and every paired per-fold difference
against YOLOv12s is consistent with zero (all p > 0.06, paired t-test over the five
folds). This is the ceiling effect the previous section described rather than a
finding about architecture: a task whose labels are whole-image and whose classes
are visually far apart does not discriminate between detectors.

The multi-class task does discriminate, and it does not favour the newest
generation. YOLOv12s places third of four on both metrics. YOLOv8s exceeds it by
2.42 points of mAP@50-95, a difference that holds in every one of the five folds
(paired p = 0.017), while YOLOv10s falls 1.98 points behind it on mAP@50
(paired p = 0.008). YOLO11s is indistinguishable from YOLOv12s on both metrics.
Three comparisons were made per metric, so under a Bonferroni correction the
YOLOv8s advantage sits at the edge of significance (0.051) and the YOLOv10s deficit
survives it (0.024). The evidence supports a small real gap, not a large one.

The cost side widens the gap. YOLOv12s needs 7.42 ms per image on the multi-class
task against 3.08 ms for YOLOv8s, so it runs 2.4 times slower while carrying 17%
fewer parameters. Its area-attention blocks buy accuracy on general benchmarks that
does not appear on this one, and they are paid for in latency that a moderation
queue would feel. A practitioner choosing a detector for gambling logo localization
on evidence from this study would choose YOLOv8s.

We continue to report YOLOv12s as the primary model because it is the model of the
study under revision, and the leakage and explanation analyses are analyses of that
system. The comparison above is what changes the recommendation, and we state it
rather than bury it: the architectural generation is not where the accuracy on this
task is decided. Section 3.4 shows where it is decided, which is the size of the
mark on the frame.

Figure 6 plots the four generations on the video-disjoint binary task.

[[FIGURE:fig_models_binary_videofold.png|Figure 6. Detector comparison on the binary task under the video-disjoint protocol.]]

Figure 7 does the same for the multi-class task, where the generations separate.

[[FIGURE:fig_models_multiclass.png|Figure 7. Detector comparison on the multi-class task.]]

### 3.4. Multi-class results and per-class reliability

Table 10 reports the five multi-class folds individually and Table 11 the
per-brand results for YOLOv12s.

**Table 10. Multi-class task per fold, YOLOv12s (%)**

[[INCLUDE:table_folds_multiclass.md]]

**Table 11. Per-class results, multi-class dataset, YOLOv12s (mean +/- SD over 5 folds)**

[[INCLUDE:table_perclass.md]]

Across five folds YOLOv12s reaches {{mc.map50}} mAP@50 on the multi-class dataset
and {{mc.map}} mAP@50-95. The two numbers describe different things, and the gap
between them is the first result worth stating plainly: identifying which brand a
mark belongs to and drawing a box that overlaps the mark by half is largely solved,
while drawing a box that holds up as the IoU requirement tightens toward 0.95 is
not. The submitted version of this paper led with the mAP@50 figure alone, which
overstated how precisely the system localizes these marks.

The per-class column shows a second gap that the aggregate hides. Brand-level
mAP@50 ranges from {{mc.worst}} on {{mc.worst.name}} to {{mc.best}} on
{{mc.best.name}}, a spread of roughly 34 points. {{mc.worst.name}} also carries the
widest fold-to-fold standard deviation in the table, so its weakness is not a
single unlucky split.

Figure 8 shows the per-class spread, ordered by mAP@50.

[[FIGURE:fig_perclass.png|Figure 8. Per-class results for YOLOv12s on the multi-class dataset, ordered by mAP@50. Error bars are one standard deviation over five folds.]]

Figure 9 shows the loss trajectories.

[[FIGURE:fig_loss_multiclass.png|Figure 9. Training and validation loss trajectories on the multi-class dataset. Shaded bands are one standard deviation across the five folds.]]

The per-class spread is not explained by sample count. BK8 has the largest number
of annotated instances in the dataset at 203, more than Starlight-Princess at 150,
and it is the weakest class by a wide margin. The size column of Table 1 explains
it instead: the median BK8 mark covers 0.76% of the image against 8.67% for
Starlight-Princess. At a 480 x 480 input, a 0.76% box spans roughly 42 pixels on a
side, and the lowest decile of BK8 boxes covers 0.07% of the image, roughly 13
pixels on a side. After three stages of downsampling, a mark that small survives as
a handful of cells on the P3 feature map and almost nothing above it. The failure
is a small-object detection failure that happens to coincide with a brand, and the
remedies are the standard ones: higher input resolution, tiled inference, or an
additional high-resolution detection head.

For a moderation deployment the consequence is specific. Recall of {{mc.bk8.r}} on
BK8 means the detector misses a substantial share of that brand's marks, so a
platform relying on this model would need to route BK8 to human review or accept
the miss rate. An aggregate mAP@50 of {{mc.map50}} does not convey that, which is
why we report the per-class table alongside it.

Table 12 reports the per-fold cost and final-epoch losses for the same runs.

**Table 12. Per-fold inference cost and final-epoch losses, multi-class task**

[[INCLUDE:table_eff_multiclass.md]]

### 3.5. Explanation quality

Table 13 reports the pointing-game and energy-based scores with both controls.

**Table 13. Explanation quality and controls**

[[INCLUDE:table_xai.md]]

The evaluation covers the multi-class dataset only. The binary annotations are
full-frame boxes, so every localization score computed against them is degenerate:
we ran it, and the real map, the random-box control, and the randomized-model check
all returned exactly 1.0, because a box covering the image contains every peak and
all of the map's energy. Those runs measure the annotation format rather than the
explanation, and we report no binary XAI numbers for that reason. This also means
the interpretability claim in the submitted version, that the heatmaps highlight
morphological markers of gambling overlays on the binary dataset, cannot be tested
with that dataset's labels.

On the multi-class dataset the maps do localize. The box-conditioned variant places
its peak inside the correct mark on {{xai.pg_box}} of images and the
prediction-independent variant on {{xai.pg_glob}}, against {{xai.pg_rand}} for a
size-matched random box and {{xai.sanity}} after the weights are re-initialized. The
separation from both controls is the result that licenses treating the maps as
evidence rather than decoration.

The energy measure is more sober than the pointing measure and we report it
because it is. {{xai.eb_glob}} of the map's mass falls inside the mark, while a
size-matched box placed at random captures {{xai.eb_rand}}. The peak lands in the
right place far more often than chance, but the bulk of the map is barely more
concentrated on the mark than on an arbitrary region of equal size. Readers should take the
pointing score as the claim being supported here and not infer a tight, mark-shaped
heatmap from it.

Figure 10 puts the two CAM variants next to both controls.

[[FIGURE:fig_xai.png|Figure 10. Pointing-game accuracy for the two CAM variants against the random-box control and the model-randomization sanity check.]]

Figure 11 shows the maps themselves on one example of each brand, so a reader can
see what the numbers in Table 13 score. The panels are illustrative and were
selected to cover the five classes rather than to represent the score
distribution; at {{xai.pg_glob}} pointing accuracy, a majority of maps do not peak
inside the mark, and the figure should be read alongside that figure rather than
in place of it.

[[FIGURE:fig_cam_multiclass.png|Figure 11. Activation CAM on the multi-class dataset, one example per brand from fold 0. Top row: the highest-confidence detection with its predicted box. Bottom row: the fused map from the three FPN scales. Panels are drawn in the 640 x 640 CAM input space.]]

Figure 12 shows the same method on the binary task and is included as the visual
form of the degeneracy argument rather than as evidence of explanation quality.
The predicted boxes cover the frame because the annotations do, and the maps
spread across the whole image with no mark to concentrate on. The second panel is
a miss, a frame carrying a gambling watermark that the model called nongambling,
and its map is indistinguishable in character from the three correct panels. No
pointing metric computed against full-frame ground truth would separate those
four cases, which is why Table 13 reports multi-class data only.

[[FIGURE:fig_cam_binary_videofold.png|Figure 12. Activation CAM on the binary task, four different Reels from the video-disjoint fold 0. Predicted boxes span the frame because every annotation in that dataset is full-frame. The second panel is a misclassification and its ground-truth class is named above it.]]

The interpretation requires care in two directions. The box-conditioned map scores
higher than the global map, and part of that advantage is structural rather than
evidential: the map is computed from channel weights taken inside the predicted
box, so when the prediction is correct the map is being scored partly on the
prediction it came from. The global column is the honest one for the question of
whether the network's features localize gambling marks independently of the
detection head's output.

The model-randomization check is what licenses any claim at all. A score that
survived weight destruction would indicate that the maps track image statistics
such as edge density or saturation rather than the trained model [52]. The
observed collapse to {{xai.sanity}} rules that out.

What the numbers do not establish is causal faithfulness. A map may localize a
mark consistently while the decision still depends on a correlated background
feature, because localization agreement and causal dependence are different
properties, and no pointing-based metric separates them. Establishing causality
would require intervention, for example occluding the marked region and measuring
the change in detection confidence. We did not run that experiment, so we describe
the maps as quantitatively validated localization evidence, and not as proof that
the model is free of background bias.

### 3.6. Cost of detection and cost of explanation

YOLOv12s runs at {{det_ms}} ms per image on an idle RTX A4000 with {{mc.params}}
parameters, measured on the multi-class task at 640 pixels. The activation CAM adds {{cam_ms}} ms, for {{total_ms}} ms end to
end.

The two numbers support different decisions. Detection at {{det_ms}} ms allows the
model to run on every upload in an ingestion pipeline. The explanation pipeline at
{{total_ms}} ms is for the review queue, where a human moderator is already the
rate-limiting step and a per-item cost in the tens of milliseconds is irrelevant.
Reporting only the combined figure would overstate the cost of screening and
understate nothing, which is why we separate them.

### 3.7. Comparison with prior work

The contribution of this work is not a higher number on the binary task. At
{{bin.map50}} mAP@50 under the random protocol the task is close to saturated, and
a further fraction of a point would carry no information. The contribution is the
pair of numbers: the same architecture, hyperparameters, and fold sizes evaluated
under two splitting rules, which isolates how much of a reported score depends on
the split rather than on the detector. To our knowledge no published study of
gambling-promotion detection in short-form video reports that pair.

Two of our results are directly usable by later work regardless of the model
chosen. First, a frame-level split of Reel-derived data leaks completely, at
{{LEAK_PCT}} of validation frames in our case, and the damage falls on the reported
variance more than on the reported mean. Second, brand-level performance on the
multi-class task is governed by mark size rather than by instance count, which
means that adding training examples of a weak brand is the wrong remedy and higher
effective resolution is the right one.

Direct numerical comparison with published gambling-detection results is not
meaningful, and we want to be explicit about why rather than present a comparison
table that implies otherwise. Those studies use different datasets, different class
definitions, and reporting conventions. Several also do not state whether frames
drawn from the same source video were kept within a single split, which is exactly
the property that Section 3.2 shows can change the apparent stability of a result.
Where that information is absent we cannot tell whether a published figure is
comparable to our random-protocol number or to our video-disjoint one, and we
prefer to say so rather than to place the figures side by side and let the reader
assume they measure the same thing.

### 3.8. Limitations

**Generalization beyond the sampled brands and platforms.** The binary dataset
draws from 300 Instagram Reels and the multi-class dataset covers five brands. The
video-disjoint result measures generalization to unseen Reels, which is a real
improvement over the random protocol, but it does not measure generalization to
unseen brands, to platforms with different compression and aspect ratios, or to
watermark styles the dataset does not contain. Operators change logos specifically
to evade detection, so the brand-level distribution shifts adversarially over time.
We did not run an external validation set, and the numbers here should not be read
as predicting performance on one.

**Small-mark detection.** Section 3.4 documents a large per-class gap driven by
mark size. At 480 x 480 the smallest decile of BK8 marks is near the resolution
floor of the architecture.

**Explanation faithfulness.** Section 3.5 establishes that the maps carry
model-dependent localization signal and not that they establish causal dependence.

**Dataset scale.** The multi-class dataset holds 443 images and 932 instances, so
per-class means over five folds carry wide intervals, visible in the standard
deviations of Table 11.

**Single hardware configuration.** Latency was measured on one GPU model. The
relative cost of detection and explanation should transfer; the absolute
milliseconds will not.
