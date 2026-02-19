# ERR@HRI 3.0: Multimodal Detection of Errors and Anticipation in Human-Robot Interactions

Official repository for the **ERR@HRI 3.0 Challenge** at [ICMI 2026](https://icmi.acm.org/2026/) (5–9 October, Napoli, Italy). Participants will find everything needed to get started: dataset access, evaluation scripts, and the baseline implementation.

> **Challenge website:** _[TBD]_
> **Contact:** _[email TBD]_

---

## Organizer TODO

- [ ] Website update and maintenance (Teresa)
- [ ] Share Call for Participation (CfP)
- [ ] Repo maintenance (Teresa)
- [ ] Organize dataset and EULA/data agreement ("participant package") (Teresa)
- [ ] Create evaluation scripts (volunteer?)
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
| Total recordings | 2,054 videos | 951 videos |
| Total duration | 36,650 s | 1,851 s |
| Temporal focus | During failure | Before failure |
| Error type | Observed failures | Anticipated outcomes |

### BAD (Bystander Affect Detection) Dataset

Webcam recordings of **45 participants'** spontaneous facial reactions while watching **46 robot and human failure scenarios** (average clip: ~17.8 s). Stimuli videos are also provided (46 `.mp4` files).

**Labels:** Binary — `Failure` (human or robot failure scenario) vs. `Control` (no failure, 6 videos).

### Bad Idea Dataset

Webcam recordings of **29 participants'** anticipatory reactions while predicting whether **30 action scenarios** would end well or poorly — captured *before* outcomes were revealed (average clip: ~1.95 s). Stimuli videos also provided (30 `.mp4` files).

**Labels:** Binary — participant's *predicted* outcome: `Well` (good outcome expected) vs. `Poorly` (bad outcome expected). Labels reflect participant prediction, not actual outcome. Data is mostly balanced (1.15 good-to-bad ratio).

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
| June 15, 2026 | Submission deadline for predictions |
| June 22, 2026 | Paper submission deadline |
| July 15, 2026 | Notification of acceptance |
| August 6, 2026 | Camera-ready papers due |
| October 5, 2026 | Challenge workshop at ICMI 2026, Napoli, Italy |

---

## Registration & Data Access

The datasets contain non-anonymized visual data. To receive access, participants must:

1. Register via the challenge website: _[link TBD]_
2. Read and sign the **EULA** (End User License Agreement)
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
