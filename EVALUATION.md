# ERR@HRI 3.0 — Evaluation Protocol

---

## Overview

Both tracks are binary classification tasks over webcam recordings. Each `(participant_id, video_id)` pair is an independent observation — participants watched each stimulus video separately, so there is no temporal continuity across videos.

Participants submit **window-level predictions** (one row per window per clip) and declare the `window_size` and `slide` they used. The organizer uses those parameters to aggregate the frame-level ground truth — since all frames within a clip share the same label, the aggregated GT is always identical to the video label.

Metrics are reported at two levels: **window** and **video**.

---

## Label Conventions

| Track | Label `0` | Label `1` |
|---|---|---|
| Track 1 (BAD) | `Control` | `Failure` |
| Track 2 (Bad Idea) | `Well` | `Poorly` |

---

## Submission Format

One CSV file per track. **One row per window** per `(participant_id, video_id)` clip.

| Column | Type | Required | Description |
|---|---|---|---|
| `participant_id` | str | ✓ | Must match the test set |
| `video_id` | str | ✓ | Must match the test set |
| `window_id` | int | ✓ | 0-indexed window position within the clip |
| `y_pred` | int | ✓ | Window-level prediction: `0` or `1` |
| `y_prob_0` | float | optional | Predicted probability for class 0 |
| `y_prob_1` | float | optional | Predicted probability for class 1 |

Participants must also **declare their window parameters** at submission time:
- `window_size`: number of frames per window
- `slide`: step size between windows

> **Note:** if `window_size=1` and `slide=1`, each window corresponds to a single frame, making this equivalent to frame-level prediction.

Submitting probability scores is optional but enables AUC-ROC.

---

## How Evaluation Works

Participants choose how to process their recordings and which window parameters to use — that is part of their method. When submitting, they declare `window_size` and `slide`. The organizer then:

1. Takes the participant's window-level predictions directly for **window-level metrics**
2. Applies majority vote across all windows per clip for **video-level metrics**
3. Joins the video-level ground-truth label (constant within each clip) onto both levels

No frame-level metrics are computed.

---

## Metrics

### Co-Primary Metrics (video level, both used for ranking)

| Metric | Why |
|---|---|
| **Macro F1** | Averages F1 equally across both classes, penalising poor performance on either |
| **Balanced Accuracy** | Mean recall per class; robust to imbalance — especially important for Track 1 (40 failure vs. 6 control videos) |

Teams are ranked by macro F1. In case of a tie, balanced accuracy is the tiebreaker. **Both metrics must be reported in the submitted paper.**

### Secondary Metrics (reported at window and video level)

| Metric | Notes |
|---|---|
| F1 (positive class) | Failure / Poorly |
| F1 (negative class) | Control / Well |
| Precision & Recall | For the positive class |
| Accuracy | |
| AUC-ROC | Only if `y_prob_0`/`y_prob_1` are submitted |

### Temporal Metrics (video level, positive-class clips only)

| Metric | Description | Better = |
|---|---|---|
| **Earliest Detection Time** | Average % of clip elapsed at the first correct window prediction, over correctly classified positive-class clips | Lower |
| **FNR per video** | Average fraction of windows that miss the positive label, over all positive-class clips | Lower |

---

## Running the Evaluator

```bash
# Track 1
python eval.py --gt ground_truth_track1.csv \
               --pred my_submission_track1.csv \
               --track 1 --window_size 5 --slide 1

# Track 2
python eval.py --gt ground_truth_track2.csv \
               --pred my_submission_track2.csv \
               --track 2 --window_size 10 --slide 2

# Save per-video breakdown to CSV
python eval.py --gt gt.csv --pred sub.csv --track 1 \
               --window_size 5 --slide 1 --out per_video.csv
```

---

## Handling Missing or Extra Predictions

- Videos in the ground truth with no submitted predictions are treated as all-zero windows.
- Videos in the submission not present in the ground truth are ignored.

Ensure your submission covers all test-set videos to avoid unnecessary penalties.

---

## Team Ranking

Teams are ranked **separately per track** by video-level macro F1, with balanced accuracy as the tiebreaker.

Award categories:
- **Track 1 Winner** — Best bystander reaction detection (BAD Dataset)
- **Track 2 Winner** — Best anticipatory response prediction (Bad Idea Dataset)
- **Best Overall Performance** — Best combined performance across both tracks
- **Best Cross-Dataset Generalization** *(optional)* — Best transfer learning result

Each team may submit **up to 3 times** per track on the test set.