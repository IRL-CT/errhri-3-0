# ERRHRI-3-0: BadNet Baseline Implementation

This repository contains the baseline implementation for the ERRHRI (Ethical Robot-Robot Human Interaction) project using BadNet architecture for robot behavior classification.

## Overview

BadNet is a convolutional neural network designed for classifying robot behavior from visual data. This implementation provides both image-based and NPY (pre-processed) data loading capabilities with comprehensive training and evaluation pipelines.

## Repository Structure

```
├── badnet/                     # Main implementation directory
│   ├── badnet_pytorch.py      # Core models and dataset classes
│   ├── train_badnet.py        # Training script with W&B integration
│   ├── get_metrics.py         # Evaluation metrics utilities
│   ├── create_image_splits.py # Data splitting utilities
│   ├── resize_dataset.py      # Dataset preprocessing
│   └── badnet.sub             # SLURM submission script
```

## Features

### Models
- **BadNetCNN**: Original BadNet architecture with configurable parameters
- **BadNetPretrained**: Transfer learning using ResNet/EfficientNet backbones
- **BadNetSimple**: Simplified architecture for faster training

### Data Loading
- **BadNetDataset**: Standard image loading with augmentation
- **BadNetDatasetWithAugmentation**: Enhanced augmentation pipeline
- **BadNetDatasetNPY**: High-performance pre-processed data loading

### Training Features
- Cross-validation with inter-participant folds
- Wandb integration for experiment tracking
- Hyperparameter sweeps
- Early stopping and model checkpointing
- Weighted loss for imbalanced datasets
- Label smoothing and regularization

## Quick Start

### 1. Environment Setup
```bash
pip install torch torchvision numpy pandas scikit-learn pillow wandb tqdm
```

### 2. Data Preparation
Organize your data in the following structure:
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

### 3. Basic Training
```bash
cd badnet
python train_badnet.py --csv_path ../../data_badidea/trainval/label_data.csv \
                       --image_base_path ../../data_badidea/trainval \
                       --epochs 100 \
                       --batch_size 32
```

### 4. NPY Data Training (Faster)
```bash
python train_badnet.py --csv_path ../../data_badidea/trainval/label_data.csv \
                       --npy_base_path ../../trainval_npy \
                       --use_npy \
                       --epochs 100
```

## Configuration Options

### Model Parameters
- `--activation`: relu, sigmoid
- `--kernel_size`: Convolution kernel size (2, 4, 6, 8)
- `--base_filters`: Base number of filters (16, 32, 64)
- `--model_type`: original, simple, pretrained_resnet18, pretrained_resnet34

### Training Parameters
- `--learning_rate`: Learning rate (0.0001, 0.001, 0.00001)
- `--batch_size`: Batch size (16, 32, 64)
- `--epochs`: Number of training epochs
- `--patience`: Early stopping patience
- `--num_folds`: Cross-validation folds

### Data Parameters
- `--use_npy`: Use pre-processed NPY files
- `--use_weighted_loss`: Handle class imbalance
- `--num_augmentations`: Data augmentation multiplier

## Performance Features

### NPY Data Format
For faster training, convert images to NPY format:
- Normalized float32 arrays (0-1 range)
- CHW format (channels-first for PyTorch)
- Significant speedup over image loading

### Caching
- In-memory image caching for datasets < 10GB
- Automatic cache management
- Progress tracking for cache building

## Evaluation Metrics

The implementation provides comprehensive metrics:
- Standard: Accuracy, Precision, Recall, F1-score
- Tolerant metrics (tolerance=1 for ordinal data)
- Cohen's Kappa for inter-rater agreement
- Confusion matrices
- Per-fold and average performance

## Experiment Tracking

Integration with Weights & Biases (wandb):
- Hyperparameter logging
- Training curves
- Model artifacts
- Sweep optimization
- Cross-validation results

## SLURM Integration

Use the provided submission script:
```bash
sbatch badnet.sub
```

## Data Format

### CSV Labels
Required columns:
- `participant_id`: Unique participant identifier
- `q_id`: Question/scenario identifier (e.g., "q_1", "q_2")
- `label`: Target classification label

### Image Naming
Expected format: `q_{id}_main_{label}_30fps_frame{number}.png`
Example: `q_6_main_1_30fps_frame0011.png`

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{errhri-badnet-2026,
  title={BadNet Implementation for Robot Behavior Classification},
  author={ERRHRI Team},
  year={2026},
  url={https://github.com/your-org/errhri-3-0}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Issues and Support

Please report issues through the GitHub issue tracker. Include:
- Error messages and stack traces
- System configuration
- Data format details
- Minimal reproducible example