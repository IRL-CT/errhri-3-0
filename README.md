# ERRHRI 3.0: Challenge Repository

This is the official repository for the **ERRHRI 3.0 Challenge** (3rd edition of the Evaluating Robot misbehavior in Robot-Human Interaction challenge). Participants will find everything needed to get started: dataset access, evaluation scripts, and the baseline implementation.

> **Challenge website:** _[link TBD]_
> **Contact:** _[email TBD]_

---

## Organizer TODO

- [ ] Website update and maintenance
- [ ] Share Call for Participation (CfP)
- [ ] Create/finalize this repository
- [ ] Organize dataset and EULA/data agreement ("participant package")
- [ ] Create evaluation scripts
- [ ] Train and document baseline
- [ ] _(later)_ Verify participants' submissions
- [ ] _(later)_ Draft challenge paper

---

## Challenge Overview

ERRHRI 3.0 is a shared task challenge focused on **detecting and classifying robot misbehavior from visual data**. Given video frames of robot-human interaction scenarios, participants develop systems that can identify whether robot behavior is appropriate or problematic — and to what degree.

This challenge builds on the BadRobotsIRL series and pushes towards more robust, generalizable methods for automated robot behavior assessment.

### Task

Given a sequence of video frames from a robot-human interaction scenario, classify the robot's behavior according to a predefined label set. Evaluation is based on frame-level predictions aggregated across participants and scenarios.

### Data

The dataset consists of video frames from recorded robot-human interaction sessions, organized by participant and scenario (`q_id`). Labels are provided in CSV format.

- **Format:** PNG frames (naming: `q_{id}_main_{label}_30fps_frame{number}.png`)
- **Splits:** Train/val and held-out test set
- **Access:** Dataset is distributed as part of the participant package — see [Registration & Data Access](#registration--data-access)

### Evaluation

Systems are evaluated using:
- Macro F1-score (primary metric)
- Tolerance-1 accuracy (for ordinal labels)
- Cohen's Kappa

Evaluation scripts will be released in this repository — see [Evaluation](#evaluation).

### Timeline

| Milestone | Date |
|---|---|
| CfP released | _TBD_ |
| Participant package available | _TBD_ |
| Evaluation scripts released | _TBD_ |
| System submission deadline | _TBD_ |
| Results announced | _TBD_ |
| Challenge paper deadline | _TBD_ |

---

## Registration & Data Access

To participate and receive the dataset:

1. Register via the challenge website: _[link TBD]_
2. Sign and return the EULA/data agreement
3. You will receive a download link for the participant package

---

## Evaluation

Evaluation scripts will be released here before the submission deadline. They will be runnable against your predictions CSV as follows:

```bash
python evaluate.py --predictions predictions.csv --labels test_labels.csv
```

_Scripts are under development — check back for updates._

---

## Baseline

This repository includes a full baseline implementation using the **BadNet** convolutional neural network architecture (PyTorch). The baseline serves as a reference point and starting template for participants.

### Baseline Results

_To be filled in once baseline training is complete._

### Repository Structure

```
├── badnet/                     # Baseline implementation
│   ├── badnet_pytorch.py      # Core models and dataset classes
│   ├── train_badnet.py        # Training script with W&B integration
│   ├── get_metrics.py         # Evaluation metrics
│   ├── create_image_splits.py # Data splitting utilities
│   ├── resize_dataset.py      # Dataset preprocessing
│   └── badnet.sub             # SLURM submission script
└── challenge_proposal.pdf     # Challenge proposal document
```

### Models

- **BadNetCNN**: Original BadNet architecture with configurable parameters
- **BadNetPretrained**: Transfer learning using ResNet/EfficientNet backbones
- **BadNetSimple**: Lightweight architecture for faster iteration

### Quick Start

**1. Environment setup**
```bash
pip install torch torchvision numpy pandas scikit-learn pillow wandb tqdm
```

**2. Data preparation**

Organize your data as follows:
```
data_badidea/
├── trainval/
│   ├── label_data.csv
│   ├── participant_1/
│   │   ├── q_1_main_0_30fps_frame0001.png
│   │   └── ...
│   └── ...
└── test/
    ├── test_label_data.csv
    └── ...
```

**3. Train the baseline**
```bash
cd badnet
python train_badnet.py --csv_path ../../data_badidea/trainval/label_data.csv \
                       --image_base_path ../../data_badidea/trainval \
                       --epochs 100 \
                       --batch_size 32
```

**4. Faster training with NPY format**

First preprocess images:
```bash
python resize_dataset.py
```

Then train:
```bash
python train_badnet.py --csv_path ../../data_badidea/trainval/label_data.csv \
                       --npy_base_path ../../trainval_npy \
                       --use_npy \
                       --epochs 100
```

### Key Training Options

| Parameter | Description | Default |
|---|---|---|
| `--model_type` | `original`, `simple`, `pretrained_resnet18`, `pretrained_resnet34` | `original` |
| `--learning_rate` | Learning rate | `0.0001` |
| `--batch_size` | Batch size | `64` |
| `--epochs` | Max training epochs | `100` |
| `--patience` | Early stopping patience | — |
| `--num_folds` | Cross-validation folds | `5` |
| `--use_npy` | Use pre-processed NPY files | `False` |
| `--use_weighted_loss` | Handle class imbalance | `False` |

### Data Format

**CSV labels** — required columns:
- `participant_id`: Unique participant identifier
- `q_id`: Scenario identifier (e.g., `q_1`, `q_2`)
- `label`: Target classification label

**Image naming:** `q_{id}_main_{label}_30fps_frame{number}.png`
Example: `q_6_main_1_30fps_frame0011.png`

### Experiment Tracking

The baseline integrates with [Weights & Biases](https://wandb.ai) for training curves, hyperparameter logging, and sweep optimization. Set your API key via environment variable before training:

```bash
export WANDB_API_KEY=your_key_here
```

### SLURM

A SLURM submission script is provided for HPC clusters:
```bash
sbatch badnet/badnet.sub
```
Edit `badnet.sub` to match your cluster configuration and set `WANDB_API_KEY` as an environment variable (do not hardcode it in the script).

---

## Citation

If you use this code or data in your research, please cite:
```bibtex
@misc{errhri3-2026,
  title={ERRHRI 3.0 Challenge},
  author={ERRHRI Team},
  year={2026},
  url={https://github.com/your-org/errhri-3-0}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Issues and Support

Please use the [GitHub issue tracker](../../issues) for questions about the baseline code. For questions about registration, data access, or the challenge rules, contact the organizers via the challenge website.
