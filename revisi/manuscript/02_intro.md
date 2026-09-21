## 1. INTRODUCTION

Indonesian regulators removed more than 5.4 million pieces of gambling-related
content between 2017 and 2024, yet promotional posts keep reappearing on
mainstream social platforms [1], [2]. The reason is a shift in tactics. Promoters
stopped writing the brand name in the caption, because caption text is exactly
what keyword filters read. They now burn the brand into the image itself: a small
logo stamped in a corner of a Reel, a watermark laid over a gameplay clip, a
sticker pasted onto a thumbnail. The text channel is clean and the visual channel
carries the advertisement. Moderation systems that read only text see nothing [3],
[4].

Object detection is the natural instrument for the visual channel, and the YOLO
family is the natural starting point because it detects and localizes in a single
forward pass at a speed that platform-scale moderation needs [23], [24], [25].
Logo and trademark detection is itself a mature line of work, with benchmarks and
methods for small, rigid, repeated marks [12], [13], [14], [15]. Recent studies
have applied YOLO variants to gambling promotion specifically and report accuracy
in the high nineties [5], [6], [7].

Those reported numbers are where the problem starts, and it is a problem of
evidence rather than of architecture. Three gaps recur across the applied
literature on this task.

The first gap is validation protocol. Gambling promotion datasets are commonly
built by sampling frames from short videos, because a single Reel yields ten
usable frames for the cost of collecting one post. If the resulting frames are
then shuffled and split at random, frames from one video land in both the training
and the validation set. The two frames differ by a fraction of a second of camera
motion, so a model can score them correctly by recognizing the video rather than
the logo. Kapoor and Narayanan surveyed 294 papers across 17 fields and found this
family of leakage to be the single most common cause of irreproducible
machine-learning claims [48]. No published study of gambling logo detection that
we are aware of reports a grouped or video-disjoint split, and none reports what
its accuracy would be under one.

The second gap is the status of the explanation. Applied detection papers
increasingly attach a class activation map to show that the model looked at the
logo [39], [40], [41]. The map is then read as proof that the decision rests on
the brand mark and not on background context. A picture of a heatmap is not a
measurement. The explainable-AI literature has supplied measurable alternatives,
including the pointing game, energy-based pointing [42], [43], and the
model-randomization sanity check of Adebayo et al. [52], which showed that several
widely used saliency methods produce visually convincing maps even after the
network weights are destroyed [45]. Applied gambling-detection work has not
adopted them.

The third gap is what a single headline number hides. Averaged mAP over a
multi-class logo set conceals that some brands are detected reliably and others
are barely detected at all, which is precisely the information a moderation team
needs when deciding which brands still require human review. Reporting an
end-to-end latency figure that folds the explanation cost into the detector cost
hides a similar decision, because a platform can run the detector on every upload
while running the explanation only on flagged items.

This study addresses the three gaps directly. Our contributions are:

1. We quantify leakage in the conventional protocol and measure its cost. Across
   the five published random folds of our binary dataset, {{LEAK_PCT}} of
   validation frames have a sibling frame from the same source Reel in training.
   Rebuilding the folds so that no Reel crosses the boundary, with the same
   2,400/600 frame sizes and the same model and hyperparameters, moves mAP@50 from
   {{BIN_MAP50}} to {{VID_MAP50}}, and widens the between-fold standard deviation on
   mAP@50-95 roughly tenfold. We also report a property of this dataset that its
   published description omits: every annotation is a full-frame box, so the task it
   defines is whole-image classification written in detection format, and its strict
   metric cannot be read as localization.
2. We report baselines instead of a single model. YOLOv8s, YOLOv10s, YOLO11s and
   YOLOv12s are trained under identical folds, hyperparameters, and seeds, so the
   comparison isolates architecture. This answers what a practitioner actually asks,
   which is whether the newest generation is worth adopting for this task.
3. We score explanations rather than illustrating them. We report the pointing
   game and energy-based pointing game for the multi-scale activation CAM, against
   a size-matched random-box control and a model-randomization sanity check, and we
   separate detection latency from the cost of the explanation pipeline.
4. We report per-class results and the failure they contain. The BK8 watermark
   reaches {{MC_WORST}} mAP@50 against {{MC_BEST}} for Starlight-Princess, and we
   trace the difference to mark size rather than to sample count.

Section 2 describes the datasets, the two validation protocols, the detectors, and
the explanation and evaluation procedures. Section 3 reports and interprets the
results, including the limitations that the numbers impose. Section 4 states the
conclusions separately for the classification and localization tasks.
