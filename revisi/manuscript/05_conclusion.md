## 4. CONCLUSION

This study trained YOLOv12s to detect gambling brand marks burned into social
media images, and evaluated it under a protocol designed to show what the
resulting numbers do and do not support.

**On the binary task**, distinguishing gambling from non-gambling frames, YOLOv12s
reaches {{bin.map50}} mAP@50 under the conventional random 5-fold split. That
figure is inflated. Every validation frame in that split has a sibling frame from
the same Instagram Reel in the training set, in all five folds. Regrouping the
folds so that no Reel crosses the boundary, changing nothing else, gives
{{vid.map50}} mAP@50 and {{vid.map}} mAP@50-95, and widens the between-fold
standard deviation on the strict metric roughly tenfold. The second pair of numbers
is the one that describes performance on Reels the model has not seen. We also
report that every annotation in this dataset is a full-frame box, which makes the
task whole-image classification expressed in detection format. Its mAP@50-95 should
not be read as a localization result, and the explanation analysis below therefore
excludes it.

**On the multi-class task**, identifying which of five gambling platforms a mark
advertises, YOLOv12s reaches {{mc.map50}} mAP@50 and {{mc.map}} mAP@50-95. The
aggregate hides a wide per-brand spread: Starlight-Princess reaches {{mc.best}}
mAP@50 while BK8 reaches {{mc.worst}} with recall of {{mc.bk8.r}}. The cause is
mark size rather than sample count, since BK8 is the most frequent class in the
dataset and also the smallest, at a median of 0.76% of image area against 8.67%
for Starlight-Princess.

**On architecture**, the newest generation is not the strongest on this task.
Trained under identical folds, hyperparameters, and seeds, the four generations are
statistically indistinguishable on the binary task, while on the multi-class task
YOLOv8s exceeds YOLOv12s by 2.42 points of mAP@50-95 in every fold and runs 2.4
times faster. A practitioner selecting a detector for this problem on the evidence
here would select YOLOv8s. We report YOLOv12s throughout because it is the system
under revision, not because it won.

**On explanation**, the multi-scale activation maps place their peak inside the
annotated mark in {{xai.pg_glob}} of cases against {{xai.pg_rand}} for a
size-matched random box, and the score falls to {{xai.sanity}} when the model
weights are randomized. The maps therefore carry localization signal that depends
on the trained model. They are qualitative interpretability evidence with a
quantitative floor established under controls. They do not demonstrate causal
faithfulness, and they do not show that the detector is free of background bias,
because a map can localize a mark consistently while the decision rests on a
correlated cue.

**On cost**, detection takes {{det_ms}} ms per image and the explanation adds
{{cam_ms}} ms, which supports running detection across an ingestion stream and
explanation only within a review queue.

We do not describe the system as robust. On unseen Reels it loses {{d.map50}}
points of mAP@50 relative to the leaked protocol, it detects the smallest brand
mark in its own training distribution poorly, and it has not been tested on brands
or platforms outside its datasets.

Four directions follow from these limitations. External validation on an
independently collected set, including gambling brands absent from training, would
measure the domain shift that Section 3.8 identifies and that these datasets cannot.
Higher input resolution, tiled inference, or an added high-resolution detection
head would target the small-mark failure that drives the per-class spread.
Intervention-based explanation evaluation, occluding the marked region and
measuring the change in detection confidence, would test the causal claim that
pointing metrics cannot reach. Finally, periodic re-evaluation against newly
collected posts is necessary for any deployment, because operators change their
marks in response to enforcement, which makes this an adversarial and
non-stationary detection problem rather than a fixed one.
