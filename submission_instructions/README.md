# ERR@HRI 3.0 — Submission Instructions

> **Submission deadline:** June 8, 2026  
> **Contact:** mb2554@cornell.edu  
> **Challenge website:** https://sites.google.com/view/errhri30/

---

## Overview

Participants submit **window-level predictions** as a CSV file — one file per track. Each row corresponds to one sliding-window prediction for a specific `(participant, video)` clip. You also declare the window parameters you used (`fps`, `window_size`, `slide`).

You may participate in Track 1, Track 2, or both. Each team may submit **up to 3 times per track**.

---

## What to Submit

### Submission CSV format

One file per track. **One row per window per clip.**

| Column | Type | Required | Description |
|---|---|---|---|
| `participant_id` | str | ✓ | Must match test set (e.g. `1179`) |
| `video_id` | str | ✓ | Must match test set (e.g. `QID106` or `q_10_main`) |
| `window_id` | int | ✓ | 0-indexed position of the window within the clip |
| `y_pred` | int | ✓ | Window-level binary prediction: `0` or `1` |
| `y_prob_0` | float | ✓ | Predicted probability for class 0 |
| `y_prob_1` | float | ✓ | Predicted probability for class 1 |

`y_prob_0 + y_prob_1` must equal 1.0 (tolerance ± 0.01).

#### Example (Track 1 — BAD Dataset)

```
participant_id,video_id,window_id,y_pred,y_prob_0,y_prob_1
1179,QID106,0,1,0.23,0.77
1179,QID106,1,1,0.18,0.82
1179,QID106,2,1,0.31,0.69
1179,QID119,0,0,0.61,0.39
...
```

#### Example (Track 2 — Bad Idea Dataset)

```
participant_id,video_id,window_id,y_pred,y_prob_0,y_prob_1
2313,q_10_main,0,0,0.72,0.28
2313,q_10_main,1,0,0.65,0.35
2313,q_11_main,0,1,0.34,0.66
...
```

### Label conventions

| Track | Label `0` | Label `1` |
|---|---|---|
| Track 1 (BAD) | Control | Failure |
| Track 2 (Bad Idea) | Well | Poorly |

---

## Window Parameters

You also declare three parameters alongside your submission file:

| Parameter | Description |
|---|---|
| `fps` | Frame rate at which you extracted features (e.g. `5`, `10`, `30`) |
| `window_size` | Number of frames per window |
| `slide` | Step size between consecutive windows (frames) |

**Constraints (enforced by the evaluator):**
- `slide ≤ window_size` — no gaps between windows allowed
- `window_size ≤ 2 × fps` — Track 1 only: max 2-second windows (e.g. at 5 fps, max 10 frames)
- Track 2 has no window size cap

If `window_size=1` and `slide=1`, each window corresponds to one frame (frame-level prediction is a valid special case).

---

## How to Validate Your Submission Before Sending

A validation script is bundled in this folder. Run it **before submitting** to catch format errors:

```bash
# Track 1 — format check only (no ground truth needed)
python submission_instructions/validate_submission.py \
    --pred my_submission_track1.csv \
    --track 1 \
    --fps 5 --window_size 10 --slide 5

# Track 2
python submission_instructions/validate_submission.py \
    --pred my_submission_track2.csv \
    --track 2 \
    --fps 30 --window_size 10 --slide 2
```

The script checks:
- All required columns are present
- `y_pred` is binary (0 or 1)
- `y_prob_0` and `y_prob_1` are in [0, 1] and sum to 1
- No duplicate `(participant_id, video_id, window_id)` rows
- `window_id` is 0-indexed and sequential per clip
- Window parameter constraints (`slide ≤ window_size`, and for Track 1: `window_size ≤ 2 × fps`)
- Coverage: all expected test-set videos are present

The validator will also warn you about any test-set videos missing from your submission. Missing videos are imputed as all-zero predictions during official evaluation, which will penalise your score.

### Expected test-set video lists

For reference, the expected `(participant_id, video_id)` pairs for each track are provided:

- `test_video_list_track1.csv` — 326 clips, 9 participants (BAD test set)
- `test_video_list_track2.csv` — 180 clips, 6 participants (Bad Idea test set)

---

## How to Submit

Send your submission files by email to **mb2554@cornell.edu** with subject line:

```
ERR@HRI 3.0 — [Team Name]
```

Include:
1. Your submission CSV file(s) — one per track (multiple ok)
2. Your declared window parameters (`fps`, `window_size`, `slide`) for each track
3. The output of `validate_submission.py` (copy-paste or attach as `.txt`) confirming your submission passed format checks
4. **A link to your code repository** (GitHub, GitLab, etc.) with instructions to reproduce your predictions — see below

### Code repository and reproducibility

Submitting your code is **required**. Your repository must include:

1. **All code** needed to go from raw test data to the submitted prediction CSV
2. A **`README`** with step-by-step instructions to reproduce your predictions, including:
   - Environment setup (e.g. `conda env create -f environment.yml` or `pip install -r requirements.txt`)
   - Any preprocessing steps (frame extraction, feature computation, etc.)
   - The exact command(s) to run inference on the test set and produce the submission CSV
3. The **exact version** of your submission (tag or commit hash) clearly identified

Your repository may be private at submission time, but you must grant access to the organizers (GitHub/GitLab username: mteresaparreira). **We will attempt to reproduce your predictions as part of the review process. Submissions that cannot be reproduced may be disqualified.**

---

## Evaluation Summary

The organizer runs `validate_submission.py` with the ground-truth labels to compute official results. Metrics differ by track:

| Track | Primary ranking metric | Tiebreaker |
|---|---|---|
| Track 1 (BAD) | Macro F1 (video level, majority vote) | Balanced Accuracy |
| Track 2 (Bad Idea) | AUC-ROC (video level, max `y_prob_1`) | — |

Both tracks also report: window-level macro F1, balanced accuracy, precision, recall, F1 per class, accuracy, AUC-ROC, earliest detection time, and FNR per video.

See [EVALUATION.md](../EVALUATION.md) for the complete evaluation protocol.

---

## Common Pitfalls

- **Missing videos:** any test clip absent from your submission is imputed as a single all-zero prediction. Always run the validator to check coverage.
- **Non-sequential `window_id`:** windows must be numbered `0, 1, 2, …` per clip with no gaps.
- **Window size cap (Track 1):** at 5 fps, the maximum `window_size` is 10. At 30 fps, the maximum is 60.
- **Slide > window_size:** this creates gaps between windows and is not allowed.
- **Wrong `video_id` format:** Track 1 uses `QID<N>` (e.g. `QID106`); Track 2 uses `q_<N>_main` (e.g. `q_10_main`). Check the expected video lists to confirm the exact strings.

---

## Paper Submission

All participating teams must submit a short paper describing their method (ICMI 2026 template). The paper must report:
- **Track 1:** video-level macro F1 and balanced accuracy
- **Track 2:** video-level AUC-ROC

Paper deadline: **June 15, 2026**. See the challenge website for submission instructions.

Code release is strongly encouraged. Accepted papers will be published in the ACM ICMI 2026 proceedings.
