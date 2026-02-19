# ERR@HRI 3.0 Baseline Implementation

This document describes the baseline implementation provided for the ERR@HRI 3.0 Challenge. The baseline uses the **BadNet** convolutional neural network architecture in PyTorch and serves as a reference point and starting template for participants.

> **Note:** This is one possible baseline approach. Participants are free to use any method.

## Baseline Results

_To be filled in by May 1, 2026._

| Track | Model | F1-score | Accuracy | AUC |
|---|---|---|---|---|
| Track 1 (BAD) | — | — | — | — |
| Track 2 (Bad Idea) | — | — | — | — |

---

## Repository Structure

```
badnet/
├── badnet_pytorch.py      # Core models and dataset classes
├── train_badnet.py        # Training script with W&B integration
├── get_metrics.py         # Evaluation metrics utilities
├── create_image_splits.py # Data splitting utilities
├── resize_dataset.py      # Dataset preprocessing (image → NPY)
└── badnet.sub             # SLURM submission script
```

---

## Models

- **BadNetCNN**: Original BadNet architecture with configurable filters, kernel size, and activation
- **BadNetPretrained**: Transfer learning using ResNet (18/34/50) or EfficientNet backbones
- **BadNetSimple**: Lightweight architecture for faster iteration

---

## Quick Start

### 1. Environment Setup

```bash
pip install torch torchvision numpy pandas scikit-learn pillow wandb tqdm
```

### 2. Data Preparation

Organize your data as follows:

```
data/
├── trainval/
│   ├── label_data.csv
│   ├── participant_1/
│   │   ├── video_001.mp4
│   │   └── ...
│   └── ...
└── test/
    ├── test_label_data.csv
    └── ...
```

**CSV labels** — required columns:
- `participant_id`: Unique participant identifier
- `q_id`: Scenario identifier (e.g., `q_1`, `q_2`)
- `label`: Target classification label

### 3. Basic Training

```bash
cd badnet
python train_badnet.py --csv_path ../data/trainval/label_data.csv \
                       --image_base_path ../data/trainval \
                       --epochs 100 \
                       --batch_size 32
```

### 4. Faster Training with NPY Format

Pre-process videos/images to NPY arrays for significantly faster data loading:

```bash
python resize_dataset.py
```

Then train using the pre-processed data:

```bash
python train_badnet.py --csv_path ../data/trainval/label_data.csv \
                       --npy_base_path ../data/trainval_npy \
                       --use_npy \
                       --epochs 100
```

### 5. SLURM (HPC clusters)

A SLURM submission script is provided:

```bash
sbatch badnet/badnet.sub
```

Edit `badnet.sub` to match your cluster configuration. **Do not hardcode API keys** — pass `WANDB_API_KEY` as an environment variable instead.

---

## Key Training Options

| Parameter | Description | Default |
|---|---|---|
| `--model_type` | `original`, `simple`, `pretrained_resnet18`, `pretrained_resnet34` | `original` |
| `--activation` | `relu`, `sigmoid` | `relu` |
| `--kernel_size` | Convolution kernel size | — |
| `--base_filters` | Base number of filters (16, 32, 64) | — |
| `--learning_rate` | Learning rate | `0.0001` |
| `--batch_size` | Batch size | `64` |
| `--epochs` | Max training epochs | `100` |
| `--patience` | Early stopping patience | — |
| `--num_folds` | Cross-validation folds | `5` |
| `--use_npy` | Use pre-processed NPY files | `False` |
| `--use_weighted_loss` | Handle class imbalance | `False` |
| `--num_augmentations` | Data augmentation multiplier | — |

---

## Evaluation Metrics

The `get_metrics.py` module computes:

- Standard: Accuracy, Precision, Recall, F1-score
- AUC/ROC (binary and multi-class)
- Windowed predictions and earliest detection time
- Per-fold and averaged performance across cross-validation splits

---

## Experiment Tracking

The baseline integrates with [Weights & Biases](https://wandb.ai) for training curves, hyperparameter logging, and sweep optimization. Set your API key via environment variable before training:

```bash
export WANDB_API_KEY=your_key_here
```

---

## Cross-Validation

The baseline uses **inter-participant** cross-validation — each fold holds out a different set of participants, evaluating generalization to unseen individuals. This mirrors the challenge's subject-independent evaluation protocol.
