#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet Robot Behavior Classification - PyTorch Implementation

Core components: models, datasets, and training utilities.
"""

import os
import re
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_image_filename(filename):
    """
    Parse image filename to extract q_id and label.
    Format: q_{id}_main_{label}_30fps_frame{number}.jpg
    Example: q_6_main_1_30fps_frame0011.jpg
    
    Returns:
        tuple: (q_id, label_from_filename, frame_number) or (None, None, None) if parsing fails
    """
    # Pattern: q_{id}_main_{label}_30fps_frame{number}
    pattern = r'(q_\d+)_main_(\d+)_30fps_frame(\d+)'
    match = re.match(pattern, filename)
    
    if match:
        q_id = match.group(1)
        label_from_filename = int(match.group(2))
        frame_number = int(match.group(3))
        return q_id, label_from_filename, frame_number
    return None, None, None


def load_label_mapping(csv_path):
    """
    Load CSV and create a mapping from (participant_id, q_id) to label.
    
    Args:
        csv_path: Path to label_data.csv
    
    Returns:
        dict: {(participant_id, q_id): label}
    """
    df = pd.read_csv(csv_path)
    
    # Create mapping - for each (participant_id, q_id), get the label (should be same for all frames)
    label_mapping = {}
    for _, row in df.iterrows():
        key = (str(row['participant_id']), row['q_id'])
        label_mapping[key] = int(row['label'])
    
    return label_mapping


class BadNetDataset(Dataset):
    """Dataset for loading robot behavior images with frame-to-label mapping."""
    
    def __init__(self, participants, image_base_path, csv_path, transform=None, cache_images=True):
        self.image_base_path = image_base_path
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Load label mapping from CSV
        self.label_mapping = load_label_mapping(csv_path)
        
        # Scan images and build samples list
        self.samples = []
        for participant in participants:
            participant_dir = os.path.join(image_base_path, participant)
            
            if not os.path.exists(participant_dir):
                print(f"Warning: Participant directory not found: {participant_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(participant_dir) if f.endswith('.png')]
            
            for img_file in image_files:
                q_id, label_from_filename, frame_num = parse_image_filename(img_file)
                
                if q_id is None:
                    print(f"Warning: Could not parse filename: {img_file}")
                    continue
                
                # Look up label from CSV
                key = (participant, q_id)
                if key in self.label_mapping:
                    label = self.label_mapping[key]
                    image_path = os.path.join(participant_dir, img_file)
                    self.samples.append((image_path, label))
                else:
                    print(f"Warning: No label found for participant {participant}, {q_id}")

        if cache_images:
            print("Caching images in memory...")
            for i, (image_path, label) in enumerate(self.samples):
                image = Image.open(image_path).convert('RGB')
                if transform:
                    image = transform(image)
                self.image_cache[image_path] = image
                if (i + 1) % 1000 == 0:
                    print(f"Cached {i + 1}/{len(self.samples)} images")

        print(f"Loaded {len(self.samples)} samples from {len(participants)} participants")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image, label


class BadNetDatasetWithAugmentation(Dataset):
    """Dataset that includes both original and augmented samples."""
    
    def __init__(self, participants, image_base_path, csv_path, num_augmentations=2):
        self.image_base_path = image_base_path
        self.num_augmentations = num_augmentations
        
        # Load label mapping from CSV
        self.label_mapping = load_label_mapping(csv_path)
        
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        
        # Scan images and build samples list
        self.samples = []
        self.labels = []
        
        for participant in participants:
            participant_dir = os.path.join(image_base_path, participant)
            
            if not os.path.exists(participant_dir):
                print(f"Warning: Participant directory not found: {participant_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(participant_dir) if f.endswith('.png')]
            
            for img_file in image_files:
                q_id, label_from_filename, frame_num = parse_image_filename(img_file)
                
                if q_id is None:
                    continue
                
                # Look up label from CSV
                key = (participant, q_id)
                if key in self.label_mapping:
                    label = self.label_mapping[key]
                    image_path = os.path.join(participant_dir, img_file)
                    
                    # Add original sample
                    self.samples.append((image_path, label, False))
                    self.labels.append(label)
                    
                    # Add augmented samples
                    for _ in range(num_augmentations):
                        self.samples.append((image_path, label, True))
                        self.labels.append(label)
        
        print(f"Dataset: {len(self.samples) // (num_augmentations + 1)} original → {len(self.samples)} total samples "
              f"({num_augmentations} augmentations per image)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label, is_augmented = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if is_augmented:
            image = self.augmentation_transform(image)
        
        image = self.base_transform(image)
        return image, label


class BadNetDatasetNPY(Dataset):
    """Dataset loading preprocessed NPY files (fastest option)."""
    
    def __init__(self, participants, image_base_path, csv_path, num_augmentations=0, cache_images=True):
        self.image_base_path = image_base_path
        self.num_augmentations = num_augmentations
        self.cache_images = cache_images
        self.image_cache = {}
        self.labels = []  # Store labels for each sample

        # Load label mapping from CSV
        self.label_mapping = load_label_mapping(csv_path)

        # Augmentation for tensor data (if needed)
        self.use_augmentation = num_augmentations > 0
        if self.use_augmentation:
            # Note: For tensor augmentation, we need different transforms
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            ])
        
        # Scan images and build samples list
        self.samples = []
        
        for participant in participants:
            participant_dir = os.path.join(image_base_path, participant)
            
            if not os.path.exists(participant_dir):
                print(f"Warning: Participant directory not found: {participant_dir}")
                continue
            
            # Get all NPY files
            npy_files = [f for f in os.listdir(participant_dir) if f.endswith('.npy')]
            
            for npy_file in npy_files:
                # Parse filename (same pattern but with .npy extension)
                # Handle both .jpg and .png source files
                png_filename = npy_file.replace('.npy', '.png')
                jpg_filename = npy_file.replace('.npy', '.jpg')
                
                # Try parsing with PNG first, then JPG
                q_id, label_from_filename, frame_num = parse_image_filename(png_filename)
                if q_id is None:
                    q_id, label_from_filename, frame_num = parse_image_filename(jpg_filename)
                
                if q_id is None:
                    continue
                
                # Look up label from CSV
                key = (participant, q_id)
                if key in self.label_mapping:
                    label = self.label_mapping[key]
                    npy_path = os.path.join(participant_dir, npy_file)
                    
                    # Add original sample
                    self.samples.append((npy_path, label, False))
                    self.labels.append(label)
                    
                    # Add augmented samples
                    for _ in range(num_augmentations):
                        self.samples.append((npy_path, label, True))
                        self.labels.append(label)

        if cache_images:
            print("Caching NPY arrays in memory...")
            unique_paths = set([s[0] for s in self.samples])
            for i, npy_path in enumerate(unique_paths):
                img_array = np.load(npy_path)  # Already normalized tensor format
                self.image_cache[npy_path] = img_array
                if (i + 1) % 1000 == 0:
                    print(f"Cached {i + 1}/{len(unique_paths)} unique arrays")
            print(f"Cached {len(self.image_cache)} unique arrays")
        
        print(f"Dataset: {len(self.samples) // (num_augmentations + 1)} original → {len(self.samples)} total samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npy_path, label, is_augmented = self.samples[idx]
        
        # Load from cache or disk
        if self.cache_images and npy_path in self.image_cache:
            img_array = self.image_cache[npy_path].copy()
        else:
            img_array = np.load(npy_path)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        # Apply augmentation if needed (limited for pre-normalized tensors)
        if is_augmented and self.use_augmentation:
            # For pre-normalized tensors, augmentation is limited
            # Simpler: just do horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])  # Flip width dimension
        
        return img_tensor, label


class BadNetCNN(nn.Module):
    """Original BADNet architecture."""
    
    def __init__(self, num_classes=2, base_filters=16, kernel_size=4, activation='relu'):
        super(BadNetCNN, self).__init__()
        
        act_fn = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            act_fn,
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_filters, base_filters*2, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            act_fn,
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            act_fn,
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_filters*4, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class BadNetPretrained(nn.Module):
    """Transfer learning model using pretrained backbones."""
    
    def __init__(self, num_classes=2, backbone='resnet18', freeze_backbone=True, dropout=0.5):
        super(BadNetPretrained, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='DEFAULT')
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            num_features = self.backbone.fc.in_features
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        if 'resnet' in backbone:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes)
            )
        elif 'efficientnet' in backbone:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


class BadNetSimple(nn.Module):
    """Simplified BADNet with fewer layers."""
    
    def __init__(self, num_classes=2, base_filters=32, dropout=0.25):
        super(BadNetSimple, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_filters, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*2*16, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(model_type='original', num_classes=2, **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type: 'original', 'simple', 'pretrained_resnet18', 'pretrained_resnet34', 
                    'pretrained_resnet50', 'pretrained_efficientnet_b0'
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Model instance
    """
    if model_type == 'original':
        # Filter kwargs for BadNetCNN
        valid_keys = ['base_filters', 'kernel_size', 'activation']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return BadNetCNN(num_classes=num_classes, **filtered_kwargs)
    
    elif model_type == 'simple':
        # Filter kwargs for BadNetSimple
        valid_keys = ['base_filters', 'dropout']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return BadNetSimple(num_classes=num_classes, **filtered_kwargs)
    
    elif model_type.startswith('pretrained_'):
        # Filter kwargs for BadNetPretrained
        valid_keys = ['freeze_backbone', 'dropout']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        backbone = model_type.replace('pretrained_', '')
        return BadNetPretrained(num_classes=num_classes, backbone=backbone, **filtered_kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_interparticipant_folds(participants, num_folds=5, seed=42):
    """
    Create inter-participant folds for cross-validation.
    Returns train/val splits only (no test set).
    
    Args:
        participants: List of participant IDs
        num_folds: Number of folds (default 5)
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples: [(train_participants, val_participants), ...]
    """
    np.random.seed(seed)
    all_participants = list(participants)
    
    print(f"Creating {num_folds} folds from {len(all_participants)} participants")
    
    np.random.shuffle(all_participants)
    
    # Distribute participants evenly across folds
    fold_size = len(all_participants) // num_folds
    remainder = len(all_participants) % num_folds
    participant_groups = []
    
    start_idx = 0
    for i in range(num_folds):
        # First 'remainder' folds get one extra participant
        current_fold_size = fold_size + (1 if i < remainder else 0)
        participant_groups.append(all_participants[start_idx:start_idx + current_fold_size])
        start_idx += current_fold_size
    
    folds = []
    for fold_idx in range(num_folds):
        # Use this fold as validation
        val_participants = participant_groups[fold_idx]
        
        # All other folds are training
        train_participants = []
        for i in range(num_folds):
            if i != fold_idx:
                train_participants.extend(participant_groups[i])
        
        folds.append((train_participants, val_participants))
        print(f"Fold {fold_idx}: Train={len(train_participants)} participants, "
              f"Val={len(val_participants)} participants")
    
    return folds


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, epochs, patience=20, checkpoint_dir='./checkpoints'):
    """Train model for one fold with early stopping."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(checkpoint_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history