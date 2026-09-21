# Response to Reviewers

**Paper ID:** 22817
**Title (revised):** Explainable detection of covert online gambling logos with YOLOv12s: leakage-controlled validation and quantitative Grad-CAM evidence
**Journal:** IAES International Journal of Artificial Intelligence (IJ-AI)

We thank the Editor-in-Chief and the three reviewers. The comments identified
problems we agree were real, and addressing them changed the paper's conclusions
rather than only its wording.

Three changes are worth flagging before the point-by-point responses, because they
affect how the rest of the paper reads.

**First, we found and corrected a validation error in our own protocol.** Reviewer
C asked us to discuss whether the very high binary accuracy might reflect dataset
similarity or source correlation. We investigated and found it does. The 3,000
binary frames were sampled ten at a time from 300 Instagram Reels, and our random
5-fold split placed frames from the same Reel on both sides of the boundary. In all
five folds, 100.0% of validation frames have a sibling frame from the same Reel in
training. We rebuilt the folds with `StratifiedGroupKFold` grouped by Reel, keeping
fold sizes and every hyperparameter identical, and retrained. The binary mAP@50
falls from {{bin.map50}} to {{vid.map50}}. Both numbers are now reported, and the
gap between them is a principal finding of the revised paper. We are grateful for
the question; it corrected a result we would otherwise have published.

**Second, we replaced illustrative explanation figures with measured ones.**
Reviewers B and C both objected to the claim that Grad-CAM proves reliance on
morphological markers. We agree the claim was not supported and have removed it. In
its place the paper reports pointing-game and energy-based pointing scores against
a size-matched random-box control and a model-randomization sanity check. We also
correct a terminology error of our own: the method uses no gradients and is an
activation-based CAM, not Grad-CAM. The title and text now say so.

**Third, we rebuilt the reference list.** It now contains 52 references, every one
with a DOI, of which two are preprints: the YOLOv12 architecture paper, which has
no peer-reviewed version and defines the model under study, and Adebayo et al.'s
sanity-checks paper, whose NeurIPS proceedings version carries no DOI. Every DOI was resolved against Crossref or
DataCite, and every citation in the body now resolves to an entry.

The point-by-point responses follow. Reviewer text is quoted; our response and the
location of the change follow each one.

---

## Editorial requirements

**"Please follow strictly the guide for authors of IJAAS."**

The manuscript was reformatted into the current IJAAS template. We also corrected a
defect we should have caught before submission: the corresponding-author block in
the submitted version contained unreplaced template boilerplate naming an author
unconnected to this work. The corresponding author is now Tanzilal Mustaqim, as
marked in the original author list, with the institutional address and email to be
supplied by the authors at resubmission. *(Front matter)*

**"Minimum 25 references, mainly journal papers."**

The list grew from 15 to 52, all with DOIs. Journal articles form the majority;
the remainder are peer-reviewed conference proceedings (CVPR, ICCV, ECCV, NeurIPS,
ICML) which are the primary venues for the detection and explainability methods the
paper builds on. *(References)*

**"Show substantial/intellectual contribution, with new findings contrasted with
existing works."**

The revision reports four contributions stated as measurable claims in Section 1:
quantified leakage in the standard protocol for this task and its cost in mAP; a
four-generation architecture comparison under identical conditions; explanation
quality measured against controls rather than illustrated; and a per-class failure
analysis traced to mark size rather than sample count. The first is, to our
knowledge, the first measurement of this effect for gambling logo detection.
*(Section 1, Sections 3.1 to 3.5)*

---

## Reviewer A

**A1. "The abstract contains several grammatical issues and awkward sentence
constructions. Improve clarity and emphasize the novelty and practical
contribution."**

The abstract was rewritten. It now opens with the operational problem, states both
dataset sizes, separates the binary and multi-class results, reports both protocols
for the binary task, and states the explanation scores with their controls. The
practical contribution is stated as the deployment implication: the video-disjoint
figure, not the random-fold figure, is what a moderation team should plan around.
*(Abstract)*

**A2. "The introduction should state the research gap and contribution more
explicitly and concisely. Distinguish your study from prior work."**

Section 1 was restructured around three named gaps: validation protocol, the status
of the explanation, and what aggregate numbers hide. Each gap is stated with what
prior work does and does not do, followed by four numbered contributions expressed
as measurable claims. *(Section 1)*

**A3. "The methodology is overly verbose in places and lacks justification for
parameter tuning choices and baseline comparison with other object-detection
models."**

Section 2 was shortened and reorganized into eight short subsections. Table 2 now
gives a rationale column for every hyperparameter and states explicitly that
settings were held constant across models by design, since per-model tuning would
confound the architecture comparison. Section 2.3 adds the baseline comparison the
reviewer asked for: YOLOv8s, YOLOv10s, YOLO11s and YOLOv12s trained under identical
folds, hyperparameters, and seeds, with results in Table 9 and Section 3.3.

The comparison changed a conclusion of the paper, so we state its outcome here rather
than leave it in a table. YOLOv12s is not the strongest detector on this task. On the
binary task the four generations are statistically indistinguishable, with every paired
per-fold difference consistent with zero. On the multi-class task YOLOv12s places third
of four: YOLOv8s exceeds it by 2.42 points of mAP@50-95 in every one of the five folds
(paired t-test, p = 0.017) and runs 2.4 times faster, at 3.08 ms per image against 7.42
ms, while carrying 17% more parameters. Section 3.3 reports this with the
Bonferroni-corrected values and states plainly that a practitioner choosing a detector
for this task on our evidence would choose YOLOv8s. We continue to report YOLOv12s as
the primary model because it is the system under revision and the object of the leakage
and explanation analyses, and the conclusion now says so explicitly rather than implying
that the newest generation won. Section 2.5 documents the paired test used.

*(Sections 2.3, 2.4,
Table 6)*

**A4. "The results and discussion are heavily descriptive. Provide deeper
analytical interpretation, comparison with previous studies, and critically discuss
limitations, especially the lower performance on small watermark classes such as
BK8 and strict localization metrics."**

Section 3 was rewritten around interpretation. On BK8 specifically, we measured
what the reviewer suspected and found the cause is not sample count: BK8 is the
most frequent class in the dataset (203 instances, more than Starlight-Princess at
150) and also the smallest, at a median of 0.76% of image area against 8.67%. At
480 x 480 the lowest decile of BK8 marks spans roughly 13 pixels, near the
resolution floor of the architecture. Table 1 now reports mark size per class, and
Section 3.4 traces the gap and names the remedies. On strict localization, Section
3.4 discusses the mAP@50-95 shortfall directly, and Section 3.8 lists it as a
limitation. Section 3.7 addresses comparison with previous studies and explains why
we do not present a numerical comparison table. *(Table 1, Sections 3.4, 3.7, 3.8)*

**A5. "Figures should be placed in text after they are mentioned in the body
text."**

Every figure and table was repositioned to follow its first mention.

**A6. "References should be formatted in IEEE style with DOI. Use Mendeley with
IEEE style."**

All 52 references are in IEEE style with DOIs. *(References)*

**A7. "Only 15 references. Enrich with at least 25 references from reputable
international journals with DOI."**

The list now holds 52. *(References)*

**A8. "Citations [17] and [20] appear in the body text but the reference list only
reaches [15]. Synchronize all citations and references."**

This was our error and it is corrected. Every in-text citation was checked against
the list. *(Throughout)*

**A9. "Six references are arXiv preprints. Reduce to a maximum of two and replace
with journal or proceedings versions."**

All six were replaced with peer-reviewed versions. Two preprints remain, which is
the maximum the reviewer allows. The first is the YOLOv12 architecture paper
(arXiv:2502.12524), which has no peer-reviewed version and defines the model under
study. The second is Adebayo et al., "Sanity Checks for Saliency Maps"
(arXiv:1810.03292), whose NeurIPS proceedings version carries no DOI and which we
cannot replace because our explanation evaluation implements its method.
*(References)*

---

## Reviewer B

**B1. "The abstract should state dataset sizes, clearly distinguish binary and
multi-class experiments, report mAP values consistently including the substantially
lower multi-class mAP@50-95 of 58.76%, and soften the Grad-CAM claim to a
qualitative observation unless quantitative XAI validation is provided."**

All four points are addressed. The abstract states 3,000 frames from 300 Reels and
443 images with 932 instances; separates the two experiments into distinct
sentences; reports mAP@50 and mAP@50-95 for both tasks, including the multi-class
mAP@50-95, which we agree should not have been omitted; and no longer claims that
the maps prove reliance on morphological markers. On the fourth point we took the
reviewer's conditional route and supplied the quantitative validation rather than
only softening the language, so the abstract reports measured scores with controls
while stating that they are evidence of localization agreement and not of causal
faithfulness. *(Abstract, Section 3.5)*

**B2. "The introduction needs a clearer research gap, an explanation of what is
novel about combining YOLOv12s with multi-scale Grad-CAM, stronger comparison with
recent logo-detection, YOLO-based and explainable object-detection methods, and
contributions expressed as measurable improvements."**

See A2 for the restructuring. On novelty we now state the claim more narrowly and
more defensibly: the combination of a detector with a multi-scale CAM is not itself
novel, and the revision does not claim it is. What the paper contributes is the
evaluation regime around it. The comparison with recent logo-detection, YOLO-based,
and explainable-detection work now draws on references [12] to [22], [23] to [38],
and [39] to [47]. Contributions are stated with numbers attached. *(Section 1)*

**B3. "Standardize outcome reporting: consistent mean and standard deviation across
folds, clearly reported evaluation thresholds and confidence settings, correct the
terminology inconsistency 'mAP@50-90' to 'mAP@50-95', emphasize class-level results
where BK8 performs much worse than Starlight-Princess, and clearly distinguish
detection latency from the computational cost of the full pipeline."**

Every point is corrected.

- Mean and standard deviation over five folds is now the reporting format in every
  results table.
- Section 2.5 is a new subsection stating the NMS IoU threshold (0.7), the maximum
  detections per image (300), that no confidence floor is applied during
  validation, and that precision, recall and F1 are reported at the F1-maximizing
  confidence. The submitted version stated an IoU threshold of 0.95, which was
  wrong; we apologize for the error.
- "mAP@50-90" was a typographical error throughout and now reads mAP@50-95. We also
  found and removed a sentence describing mAP@50-95 as "very low at 99.47%", which
  was incoherent.
- Class-level results are now emphasized in Table 11 and analyzed in Section 3.4,
  including the operational consequence of BK8 recall for a review queue.
- Section 3.6 reports detection latency and explanation cost as separate quantities
  and explains why the distinction affects deployment.

*(Sections 2.5, 3.4, 3.6, Tables 5 to 7)*

---

## Reviewer C

**C1. "The discussion needs more critical interpretation of generalization,
limitations, XAI reliability and practical deployment, comparison with recent
logo/object-detection studies, and a critical discussion of the exceptionally high
binary performance, particularly possible dataset similarity or source correlation
and its effect on generalization."**

This comment led to the largest change in the paper. We tested the hypothesis and
it was correct: the binary dataset's frames come ten per Reel, and the random split
leaked 100.0% of validation frames in every fold. The retrained video-disjoint
result is {{vid.map50}} mAP@50 against {{bin.map50}}. Sections 3.1 and 3.2 report
the measurement and the cost; Section 3.8 discusses what the video-disjoint number
still does not establish, namely generalization to unseen brands and platforms.
*(Sections 3.1, 3.2, 3.7, 3.8)*

**C2. "Acknowledge that Grad-CAM heatmaps provide qualitative evidence and do not
establish causal faithfulness or eliminate background bias."**

Acknowledged in the abstract, in Section 3.5, and in the conclusion. Section 3.5
states that a map can localize a mark consistently while the decision depends on a
correlated background cue, that no pointing-based metric separates the two, and
that testing the causal claim would require intervention experiments we did not
run. *(Section 3.5, Section 4)*

**C3. "Discuss domain-shift limitations including unseen gambling brands, different
platforms, image quality, compression, and watermark styles."**

Section 3.8 opens with this. We add the point that the shift here is adversarial,
since operators change marks in response to enforcement, so the problem is
non-stationary and periodic re-evaluation is a deployment requirement rather than
future work. *(Section 3.8, Section 4)*

**Change of title and framing.**

Following the finding reported immediately below, the title is changed from
"Explainable AI for Covert Online Gambling Detection Using YOLOv12 and Grad-CAM" to
"Classifying and localizing covert online gambling promotion with YOLOv12s:
leakage-controlled validation and quantitative explanation evidence." The submitted
title described both experiments as detection. Only one of them is. The new title
names the two tasks separately and signals the two methodological contributions the
reviewers asked us to make explicit. *(Title, Abstract, Section 2.1, Section 3)*

**Additional finding not raised by the reviewers, reported here because it bears
directly on C2.**

While building the quantitative XAI evaluation we found that every annotation in
the binary Instagram dataset is a full-frame box (`0.5 0.5 1.0 1.0`), one per image,
across all 3,000 frames. The binary task is therefore whole-image classification
expressed in detection format. This has three consequences we now state in the
paper rather than leave for a reader to discover.

First, the binary mAP@50-95 reported in the submitted version is not a measure of
localization, because a predicted full-frame box scores a near-unity IoU by
construction. Second, it is the first-order explanation for the exceptionally high
binary performance that Reviewer C asked us to account for, ahead of the leakage:
deciding whether a frame carries a gambling overlay is a much easier problem than
locating the overlay, and the binary dataset only ever posed the former. Third, the
interpretability claim in the submitted abstract, that the heatmaps highlight
morphological markers of gambling overlays, cannot be tested against that dataset's
labels at all. We ran the localization scores on it and obtained exactly 1.0 for the
real map, for the size-matched random-box control, and for the randomized-model
sanity check alike, which is what a full-frame ground truth forces. All reported XAI
numbers are therefore computed on the multi-class dataset only, where the median
annotated mark covers 3.1% of the image. *(Abstract, Section 3.2, Section 3.5,
Section 4)*

**C4. "The conclusion should summarize binary and multi-class results separately,
avoid calling the system robust without acknowledging limitations, describe
Grad-CAM as qualitative interpretability evidence rather than definitive proof of
unbiasedness, and include future work on external validation, unseen-brand testing,
improved small-object detection, and quantitative XAI evaluation."**

Section 4 was rewritten to this structure: separate paragraphs for the binary task
(both protocols), the multi-class task (with the per-class spread), explanation,
and cost. The word "robust" is removed, replaced by a paragraph stating the three
specific reasons we do not use it. The four future-work directions the reviewer
lists are all present, with the quantitative XAI item narrowed to
intervention-based evaluation since the pointing-based part is now in the paper
itself. *(Section 4)*

---

## Summary of changes

| Item | Submitted | Revised |
|---|---|---|
| References | 15, 6 preprints, 2 dangling citations | 52, all with DOI, 2 preprints, no dangling citations, no uncited entries |
| Binary validation | Random 5-fold only | Random and video-disjoint 5-fold, leakage quantified |
| Binary mAP@50 | {{bin.map50}} | {{bin.map50}} random, {{vid.map50}} video-disjoint |
| Baseline models | YOLOv12s only | YOLOv8s, YOLOv10s, YOLO11s, YOLOv12s, identical conditions |
| Explanation evidence | Heatmap figures | Pointing game, energy pointing, random-box control, model-randomization check |
| Explanation method name | "Grad-CAM" | Activation-based CAM, corrected |
| Latency | Single combined figure | Detection and explanation reported separately |
| Per-class analysis | Table only | Table plus cause analysis traced to mark size (Table 1) |
| Evaluation thresholds | IoU stated as 0.95, incorrect | NMS IoU 0.7, max 300 detections, no confidence floor |
| Terminology | "mAP@50-90"; mAP@50-95 called "very low at 99.47%" | Corrected throughout |
| Corresponding author | Template boilerplate | Corrected |
| Required IJAAS sections | Absent | Funding, CRediT contributions, conflict of interest, data availability, ethics |
