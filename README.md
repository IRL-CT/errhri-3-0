# ERR@HRI 3.0: Multimodal Detection of Errors and Anticipation in Human-Robot Interactions

Official repository for the **ERR@HRI 3.0 Challenge** at [ICMI 2026](https://icmi.acm.org/2026/) (5–9 October, Napoli, Italy). Participants will find everything needed to get started: dataset access, evaluation scripts, and the baseline implementation.

> **Challenge website:** [here](https://sites.google.com/view/errhri30/)
> **Contact:** Maria Teresa Parreira (mb2554@cornell.edu)

---

## Organizer TODO

- [x] Website update and maintenance (Teresa)
- [ ] Share Call for Participation (CfP)
- [x] Repo maintenance (Teresa)
- [x] Organize dataset and data agreement ("participant package") (Teresa)
- [x] Create evaluation scripts (volunteer?)
- [ ] Train and document baseline (volunteer?)
- [ ] _(later)_ Verify participants' submissions
- [ ] _(later)_ Draft challenge paper

---

## Challenge Overview

ERR@HRI 3.0 addresses the problem of **multimodal error detection in human-robot interaction** by providing two complementary datasets that span the temporal spectrum of error management — from anticipatory responses *before* failures occur to reactive responses *during* observed errors.

Building on ERR@HRI 2024 (ICMI'24) and ERR@HRI 2.0 (ACM MM'25), this third edition takes a key step forward: rather than pre-extracted features, participants receive **raw, non-anonymized video data** collected in naturalistic, crowdsourced settings. This enables end-to-end learning approaches and exposes models to the variability characteristic of real-world deployments (diverse lighting, camera angles, participant positioning, and environmental contexts).

### Challenge Goals

- Advance multimodal error detection through end-to-end learning from raw visual data
- Promote development of robust models that handle naturalistic data variability
- Explore the relationship between anticipatory and reactive error responses
- Establish benchmarks for generalizable error detection across contexts and temporal stages
- Foster community development of open, reproducible error detection methods

---

## Datasets

ERR@HRI 3.0 provides two complementary datasets, both collected via crowdsourcing (Prolific) as raw audiovisual webcam recordings (`.mp4`). Neither dataset is anonymized.

| Characteristic | BAD Dataset | Bad Idea Dataset |
|---|---|---|
| Participants | 45 | 29 |
| Total recordings | 1,645 videos | 865 videos |
| Total duration | 25,527 s | 1,851 s |
| Frame rate | 30 fps | 30 fps |
| Temporal focus | During failure | Before failure |
| Error type | Observed failures | Anticipated outcomes |

### BAD (Bystander Affect Detection) Dataset

Webcam recordings of **45 participants'** spontaneous facial reactions while watching **46 robot and human failure scenarios** (average clip: ~15.5 s at 30 fps). Stimuli videos are also provided (46 `.mp4` files).

**Labels:** Binary — `1` (Failure: human or robot failure scenario) vs. `0` (Control: no failure, 6 videos per participant). Labels are encoded in the video filename as `QID<stimulus_id>_<label>.mp4`.

#### Data Splits

| Split | Participants | Total videos | Label 0 (Control) | Label 1 (Failure) | Total Duration |
|---|---|---|---|---|---|
| trainval | 36 | 1,319 | 173 (13.1%) | 1,146 (86.9%) | 20,567 s |
| test | 9 | 326 | 43 (13.2%) | 283 (86.8%) | 4,960 s |
| **Total** | **45** | **1,645** | **216 (13.1%)** | **1,429 (86.9%)** | **25,527 s** |

Detailed per-video statistics (participant, split, video name, label, and duration in seconds) are provided in [`baddataset_stats.csv`](baddataset_stats.csv). Video names follow the convention `QID<stimulus_id>_<label>.mp4`, where `<stimulus_id>` identifies the stimulus scenario and `<label>` is the binary label (0 = Control, 1 = Failure).

#### Per-Participant Statistics

| Participant | Split | Videos | Label 0 (Control) | Label 1 (Failure) | Duration (s) |
|---|---|---|---|---|---|
| 1001 | trainval | 31 | 5 | 26 | 519.3 |
| 1115 | trainval | 37 | 5 | 32 | 595.9 |
| 1290 | trainval | 43 | 5 | 38 | 597.3 |
| 1402 | trainval | 38 | 5 | 33 | 605.4 |
| 1447 | trainval | 45 | 5 | 40 | 641.2 |
| 1674 | trainval | 32 | 5 | 27 | 520.9 |
| 2144 | trainval | 31 | 5 | 26 | 508.0 |
| 2206 | trainval | 36 | 5 | 31 | 600.0 |
| 2485 | trainval | 30 | 5 | 25 | 531.2 |
| 2568 | trainval | 37 | 5 | 32 | 595.1 |
| 2615 | trainval | 44 | 5 | 39 | 619.8 |
| 2733 | trainval | 37 | 5 | 32 | 605.3 |
| 2767 | trainval | 20 | 5 | 15 | 344.1 |
| 3367 | trainval | 27 | 5 | 22 | 414.8 |
| 3537 | trainval | 26 | 5 | 21 | 476.9 |
| 4195 | trainval | 39 | 4 | 35 | 630.8 |
| 4297 | trainval | 41 | 5 | 36 | 625.1 |
| 4456 | trainval | 45 | 5 | 40 | 589.6 |
| 4667 | trainval | 37 | 5 | 32 | 590.6 |
| 4806 | trainval | 22 | 3 | 19 | 365.6 |
| 4807 | trainval | 44 | 5 | 39 | 616.5 |
| 4928 | trainval | 30 | 5 | 25 | 517.1 |
| 5141 | trainval | 34 | 3 | 31 | 528.8 |
| 5230 | trainval | 38 | 5 | 33 | 611.8 |
| 5409 | trainval | 37 | 5 | 32 | 575.4 |
| 5729 | trainval | 45 | 5 | 40 | 621.5 |
| 6247 | trainval | 36 | 3 | 33 | 630.5 |
| 6682 | trainval | 37 | 5 | 32 | 622.3 |
| 6778 | trainval | 45 | 5 | 40 | 630.4 |
| 7446 | trainval | 41 | 5 | 36 | 616.6 |
| 7675 | trainval | 45 | 5 | 40 | 639.9 |
| 7694 | trainval | 37 | 5 | 32 | 590.6 |
| 7749 | trainval | 39 | 5 | 34 | 609.1 |
| 9009 | trainval | 38 | 5 | 33 | 621.2 |
| 9483 | trainval | 33 | 5 | 28 | 504.3 |
| 9922 | trainval | 42 | 5 | 37 | 654.0 |
| 1179 | test | 30 | 5 | 25 | 491.0 |
| 1769 | test | 40 | 5 | 35 | 630.6 |
| 2505 | test | 34 | 5 | 29 | 544.8 |
| 3272 | test | 45 | 5 | 40 | 627.8 |
| 4302 | test | 45 | 5 | 40 | 578.3 |
| 4891 | test | 37 | 5 | 32 | 590.0 |
| 6095 | test | 37 | 5 | 32 | 574.8 |
| 7209 | test | 21 | 3 | 18 | 315.6 |
| 8564 | test | 37 | 5 | 32 | 607.4 |

### Bad Idea Dataset

Webcam recordings of **29 participants'** anticipatory reactions while predicting whether **30 action scenarios** would end well or poorly — captured *before* outcomes were revealed (average clip: ~1.95 s). Stimuli videos also provided (30 `.mp4` files).

**Labels:** Binary — participant's *predicted* outcome: `1` (good outcome expected, i.e. "Well") vs. `0` (bad outcome expected, i.e. "Poorly"). Labels reflect participant prediction, not actual outcome.

#### Data Splits

| Split | Participants | Total videos | Label 0 (Poorly) | Label 1 (Well) |
|---|---|---|---|---|
| trainval | 23 | 685 | 360 (52.6%) | 325 (47.4%) |
| test | 6 | 180 | 103 (57.2%) | 77 (42.8%) |
| **Total** | **29** | **865** | **463 (53.5%)** | **402 (46.5%)** |

Note: participant 1483 is missing one video (q_20_main); participant 9055 is missing one video (q_12_main). All other participants have 30 videos each.

#### Per-Participant Statistics

Detailed per-video statistics (participant, split, video name, label, and frame count) are provided in [`badidea_dataset_stats.csv`](badidea_dataset_stats.csv). Frame names follow the convention `q_ID_main_LABEL_30fps_frameNNNN.png`, where `q_ID_main` is the video identifier, `LABEL` is the binary label (0 or 1), and `NNNN` is the zero-padded frame index.

Summary per participant:

| Participant | Split | Videos | Label 0 | Label 1 |
|---|---|---|---|---|
| 1048 | trainval | 30 | 19 | 11 |
| 1251 | trainval | 30 | 18 | 12 |
| 1483 | trainval | 28 | 15 | 13 |
| 1676 | trainval | 30 | 21 | 9 |
| 2103 | trainval | 30 | 11 | 19 |
| 2698 | trainval | 30 | 16 | 14 |
| 2946 | trainval | 30 | 14 | 16 |
| 3157 | trainval | 30 | 12 | 18 |
| 3203 | trainval | 30 | 15 | 15 |
| 3339 | trainval | 30 | 17 | 13 |
| 3882 | trainval | 30 | 15 | 15 |
| 5099 | trainval | 30 | 16 | 14 |
| 5124 | trainval | 30 | 19 | 11 |
| 5233 | trainval | 30 | 12 | 18 |
| 5310 | trainval | 30 | 12 | 18 |
| 7136 | trainval | 30 | 17 | 13 |
| 7797 | trainval | 30 | 16 | 14 |
| 8184 | trainval | 30 | 21 | 9 |
| 8758 | trainval | 30 | 12 | 18 |
| 9055 | trainval | 27 | 13 | 14 |
| 9385 | trainval | 30 | 16 | 14 |
| 9777 | trainval | 30 | 16 | 14 |
| 9941 | trainval | 30 | 17 | 13 |
| 2313 | test | 30 | 17 | 13 |
| 5009 | test | 30 | 16 | 14 |
| 6488 | test | 30 | 17 | 13 |
| 7782 | test | 30 | 24 | 6 |
| 8436 | test | 30 | 15 | 15 |
| 8786 | test | 30 | 14 | 16 |

---

## Challenge Tracks

### Track 1: Bystander Reaction Detection (BAD Dataset)
Binary classification of whether the participant is observing a **failure vs. control** scenario.

### Track 2: Anticipatory Response Prediction (Bad Idea Dataset)
Binary classification of participants' **predicted outcome** (well/poorly) from their anticipatory facial behavior.

### Cross-Dataset Generalization _(Optional)_
Participants are encouraged to explore transfer learning across datasets (e.g., train on BAD, test on Bad Idea). Evaluated separately with its own award.

---

## Evaluation

Models are evaluated on multiple metrics:

- **Offline:** F1-score (primary), Accuracy, AUC
- **Windowed predictions:** Fixed-length sliding windows; a video is correctly classified if *any* window predicts the correct label
- **Earliest Detection Time:** For correctly classified videos, percentage of video elapsed before first correct prediction (lower is better)
- **False Negative Rate per video:** Count of windows that "miss" positive predictions in error/bad-outcome videos

Teams are ranked separately per track based on F1-score. Winners selected for: Track 1, Track 2, Best Overall Performance, and Best Cross-Dataset Generalization.

Evaluation scripts will be released in this repository by **May 1, 2026**.

---

## Timeline

| Date | Milestone |
|---|---|
| March 1, 2026 | Challenge announcement and call for participation |
| March 15, 2026 | Registration opens; training and validation data released |
| May 1, 2026 | Baseline models and evaluation scripts released |
| June 1, 2026 | Test data released (without labels) |
| June 8, 2026 | Submission deadline for predictions |
| June 15, 2026 | Paper submission deadline |
| July 8, 2026 | Notification of acceptance |
| July 23, 2026 | Camera-ready papers due (Hard Deadline) |
| October 5, 2026 | Challenge workshop at ICMI 2026, Napoli, Italy |

---

## Registration & Data Access

The datasets contain non-anonymized visual data. To receive access, participants must:

1. Register via the challenge website: _[]_
2. Read and sign the **DUA** (Data Use Agreement)
3. Receive dataset download link

**Data usage terms include:** no redistribution rights; datasets may only be used for this challenge; no use to defame participants; proper data security measures required.

Each team may submit up to **3 times** on the test set. Participating teams must submit a short paper describing their approach (ICMI 2026 template). Code release is strongly encouraged. All accepted papers will be published in the ACM ICMI 2026 proceedings.

---

## Baseline

A baseline implementation is provided in the [`badnet/`](badnet/) directory. See [BASELINE.md](BASELINE.md) for full documentation, quick-start instructions, and results.

_Baseline results to be added by May 1, 2026._

---

## Repository Structure

```
├── badnet/                     # Baseline implementation
│   ├── badnet_pytorch.py      # Core models and dataset classes
│   ├── train_badnet.py        # Training script
│   ├── get_metrics.py         # Evaluation metrics
│   ├── create_image_splits.py # Data splitting utilities
│   ├── resize_dataset.py      # Dataset preprocessing
│   └── badnet.sub             # SLURM submission script
├── baddataset_stats.csv       # Per-video stats for BAD dataset (participant, split, video name, label, duration)
├── badidea_dataset_stats.csv  # Per-video stats for Bad Idea dataset (participant, split, video name, label, frame count)
├── BASELINE.md                # Baseline documentation
├── challenge_proposal.pdf     # Challenge proposal (Parreira et al., ICMI'26)
└── LICENSE
```

---

## Organizers

- Maria Teresa Parreira — Cornell University, USA
- Micol Spitale — Politecnico di Milano, Italy
- Maia Stiber — Microsoft Research, USA
- Shiye Cao — Johns Hopkins University, USA
- Amama Mahmood — Johns Hopkins University, USA
- Chien-Ming Huang — Johns Hopkins University, USA
- Hatice Gunes — University of Cambridge, UK
- Wendy Ju — Cornell Tech, USA

---

## Citation

If you use the datasets or code from this challenge, please cite:

```bibtex
@inproceedings{parreira2026errhri,
  title={ERR@HRI 3.0 Challenge: Multimodal Detection of Errors and Anticipation in Human-Robot Interactions},
  author={Parreira, Maria Teresa and Spitale, Micol and Stiber, Maia and Cao, Shiye and Mahmood, Amama and Huang, Chien-Ming and Gunes, Hatice and Ju, Wendy},
  booktitle={Proceedings of the 28th ACM International Conference on Multimodal Interaction (ICMI '26)},
  year={2026},
  publisher={ACM}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Issues and Support

For questions about the baseline code, please use the [GitHub issue tracker](../../issues). For questions about registration, data access, or challenge rules, contact the organizers via the challenge website.
