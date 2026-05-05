# ERR@HRI 3.0 Baseline Implementation

This document describes the baseline implementation provided for the ERR@HRI 3.0 Challenge. The baseline uses the **BadNet** convolutional neural network architecture in PyTorch and serves as a reference point and starting template for participants.

> **Note:** This is one possible baseline approach. Participants are free to use any method.

## Baseline Results

Performance is reported on the **held-out test set**. Primary ranking metric is **macro F1 at video level** (majority vote across windows).

| Track | Model | Backbone | F1-macro (vid) | Balanced Acc (vid) | AUC (vid) | Det. Time |
|---|---|---|---|---|---|---|
| Track 1 (BAD) | BadNetCNN | — | **0.502** | 0.504 | 0.554 | 8.8% |
| Track 2 (Bad Idea) | BadNetPretrained | ResNet-34 | **0.561** | 0.572 | 0.550 | 35.6% |

### Track 1 — Baseline Model Configuration

| Hyperparameter | Value |
|---|---|
| Architecture | `original` (BadNetCNN) |
| Activation | sigmoid |
| Kernel size | 8 |
| Base filters | 64 |
| Freeze backbone | No |
| Dropout | 0.7 |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 350 |
| Weighted loss | Yes |
| Augmentations | 2× per frame |
| Window size | 5 frames |
| Slide | 2 frames |
| Seed | 1369 |

Full results (video level, window level, and temporal metrics) and pretrained weights are available in [`baseline/models/bad/`](baseline/models/bad/).

### Track 2 — Baseline Model Configuration

| Hyperparameter | Value |
|---|---|
| Architecture | `pretrained_resnet34` |
| Freeze backbone | No |
| Dropout | 0.7 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 100 |
| Weighted loss | Yes |
| Augmentations | 3× per frame |
| Window size | 10 frames |
| Slide | 2 frames |
| Seed | 1369 |

Full results (video level, window level, and temporal metrics) and pretrained weights are available in [`baseline/models/badidea/`](baseline/models/badidea/).

---

## Repository Structure

```
baseline/
├── badnet_pytorch.py      # Core models and dataset classes
├── train_badnet.py        # Cross-validation training script with W&B integration
├── train_final.py         # Train on full trainval set + generate test predictions
├── get_metrics.py         # Evaluation metrics utilities
├── create_image_splits.py # Data splitting utilities
├── resize_dataset.py      # Dataset preprocessing (image → NPY)
└── models/
    ├── bad/
    │   ├── model.pt              # Pretrained weights (Track 1 baseline)
    │   ├── model_config.json     # Full hyperparameter configuration
    │   ├── training_history.json # Per-epoch training loss and accuracy
    │   └── results.json          # Full evaluation results on the test set
    └── badidea/
        ├── model.pt              # Pretrained weights (Track 2 baseline)
        ├── model_config.json     # Full hyperparameter configuration
        ├── training_history.json # Per-epoch training loss and accuracy
        └── results.json          # Full evaluation results on the test set
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

### 2. Frame Extraction (BAD Dataset only)

The BAD dataset is provided as raw `.mp4` files. Extract frames before training:

```bash
python utils/extract_frames.py --split trainval
python utils/extract_frames.py --split test
```

This extracts frames at **5 fps** and produces both PNG frames and NPY arrays, along with `label_data.csv` in each output directory. The Bad Idea dataset is already provided as pre-extracted frames — skip this step for Track 2.

### 3. Data Layout

After extraction, your data should be organised as:

```
<dataset_dir>/
├── trainval_frames/           # PNG frames
│   ├── label_data.csv
│   └── <participant_id>/
│       └── q_<id>_main_<label>_5fps_frame<NNNN>.png
└── trainval_npy/              # pre-processed NPY arrays
    ├── label_data.csv
    └── <participant_id>/
        └── q_<id>_main_<label>_5fps_frame<NNNN>.npy
```

**`label_data.csv` columns:**
- `participant_id`: Unique participant identifier
- `q_id`: Scenario identifier (e.g., `q_1`, `q_2`)
- `label`: Target classification label

> **Important:** `label_data.csv` contains **one row per frame**, not one row per video. All frames from the same video share the same label. The dataset classes in `badnet_pytorch.py` index individual frames during training.

### 4. Basic Training

```bash
cd baseline
python train_badnet.py --csv_path <dataset_dir>/trainval_frames/label_data.csv \
                       --image_base_path <dataset_dir>/trainval_frames \
                       --epochs 100 \
                       --batch_size 32
```

### 5. Faster Training with NPY Format

If you extracted NPY arrays in step 2 (or ran `resize_dataset.py` separately), train on the pre-processed data for significantly faster loading:

```bash
cd baseline
python train_badnet.py --csv_path <dataset_dir>/trainval_npy/label_data.csv \
                       --npy_base_path <dataset_dir>/trainval_npy \
                       --use_npy \
                       --epochs 100
```

### 6. Reproducing the Baseline Models

To train the Track 1 baseline configuration from scratch using cross-validation:

```bash
cd baseline
python train_badnet.py \
    --csv_path       <dataset_dir>/trainval_npy/label_data.csv \
    --npy_base_path  <dataset_dir>/trainval_npy \
    --use_npy --cache_images \
    --model_type     original \
    --activation     sigmoid \
    --kernel_size    8 \
    --base_filters   64 \
    --dropout        0.7 \
    --batch_size     32 \
    --epochs         350 \
    --learning_rate  0.0001 \
    --use_weighted_loss \
    --num_augmentations 2 \
    --window_size    5 \
    --slide_length   2 \
    --seed           1369
```

To train the Track 2 baseline configuration from scratch using cross-validation:

```bash
cd baseline
python train_badnet.py \
    --csv_path       <dataset_dir>/trainval_npy/label_data.csv \
    --npy_base_path  <dataset_dir>/trainval_npy \
    --use_npy --cache_images \
    --model_type     pretrained_resnet34 \
    --dropout        0.7 \
    --batch_size     64 \
    --epochs         100 \
    --learning_rate  0.001 \
    --use_weighted_loss \
    --num_augmentations 3 \
    --window_size    10 \
    --slide_length   2 \
    --seed           1369
```

### 7. Training on the Full Trainval Set

`train_final.py` trains on all trainval participants (no held-out fold) and generates a ready-to-submit prediction file for the test set.

**Track 1 (BAD dataset):**

```bash
cd baseline
python train_final.py \
    --run_name       my_model \
    --output_dir     ./runs/my_model \
    --track          1 \
    --csv_path       <dataset_dir>/trainval_npy/label_data.csv \
    --npy_base_path  <dataset_dir>/trainval_npy \
    --test_npy_base_path <dataset_dir>/test_npy \
    --test_csv_path  <dataset_dir>/test_npy/label_data.csv \
    --use_npy --cache_images \
    --model_type     original \
    --activation     sigmoid \
    --kernel_size    8 \
    --base_filters   64 \
    --dropout        0.7 \
    --batch_size     32 \
    --epochs         350 \
    --learning_rate  0.0001 \
    --use_weighted_loss \
    --num_augmentations 2 \
    --window_size    5 \
    --slide_length   2 \
    --fps            5 \
    --seed           1369
```

**Track 2 (Bad Idea dataset):**

```bash
cd baseline
python train_final.py \
    --run_name       my_model \
    --output_dir     ./runs/my_model \
    --track          2 \
    --csv_path       <dataset_dir>/trainval_npy/label_data.csv \
    --npy_base_path  <dataset_dir>/trainval_npy \
    --test_npy_base_path <dataset_dir>/test_npy \
    --use_npy --cache_images \
    --model_type     pretrained_resnet34 \
    --dropout        0.7 \
    --batch_size     64 \
    --epochs         100 \
    --learning_rate  0.001 \
    --use_weighted_loss \
    --num_augmentations 3 \
    --window_size    10 \
    --slide_length   2 \
    --fps            30 \
    --seed           1369
```

Both commands save `model.pt`, `config.json`, `test_predictions.csv` (window-level, ready for `eval.py`), and `eval_results.json` to `--output_dir`.

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
