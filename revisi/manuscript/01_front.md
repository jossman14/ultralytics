# Classifying and localizing covert online gambling promotion with YOLOv12s: leakage-controlled validation and quantitative explanation evidence

Tanzilal Mustaqim(1*), Pima Hani Safitri(2), Yasinta Romadhona(3)

(1,2,3) Informatics Study Program, Telkom University, Surabaya Campus, Surabaya, Indonesia

(1,2,3) Center of Excellence for Motion Technology for Safety Health and Wellness, Research Institute of Sustainable Society, Telkom University, Surabaya, Indonesia

**Corresponding Author:**

Tanzilal Mustaqim

Informatics Study Program, Telkom University, Surabaya Campus

[AUTHORS TO COMPLETE: street address, city, postal code, Indonesia]

Email: [AUTHORS TO COMPLETE]

## ABSTRACT

Operators of illegal gambling sites stamp brand logos onto social media frames, out of
reach of keyword filters. Detectors for this task are usually reported with one accuracy
figure and a few heatmaps, which tells a moderator neither how the system behaves on
unseen posts nor whether the explanations mean anything. We separate two
tasks usually reported as one: whole-image classification of gambling promotion, on 3,000 Instagram frames from
300 Reels annotated with full-frame boxes, and logo localization, on 932 logos across
five classes with a median area of 3.1%. Against YOLOv8s, YOLOv10s and YOLO11s under identical folds and seeds, YOLOv12s places third on localization: YOLOv8s beats it by 2.42 points of mAP@50-95 in every fold and runs 2.4 times faster. A random 5-fold split gives the binary
task {{BIN_MAP50}} mAP@50, yet every validation frame has a sibling from the same Reel in
training. Regrouping so no Reel crosses the boundary gives {{VID_MAP50}} and widens the
between-fold standard deviation on the strict metric roughly tenfold. Because those boxes
are full-frame, that task's mAP@50-95 measures classification, not localization. Multi-class mAP@50 is {{MC_MAP50}}, ranging
per class from {{MC_WORST}} on BK8 to {{MC_BEST}} on Starlight-Princess, a gap governed by
mark size rather than instance count. Explanations are scored on multi-class data only:
maps peak inside the logo in {{PG_GLOB}} of cases against {{PG_RAND}} for a random box,
collapsing to {{PG_SANITY}} under weight randomization. That is localization agreement,
not causal faithfulness.

**Keywords:** YOLOv12s; online gambling detection; explainable AI; data leakage;
class activation mapping; visual content moderation
