#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inter-participant data splits for image classification.

This module provides utilities for creating train/val splits
based on participant grouping (inter-participant cross-validation).
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


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
    
    # Create mapping
    label_mapping = {}
    for _, row in df.iterrows():
        key = (str(row['participant_id']), row['q_id'])
        label_mapping[key] = int(row['label'])
    
    return label_mapping


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
    
    # Shuffle participants
    np.random.shuffle(all_participants)
    
    # Split participants into num_folds groups
    fold_size = len(all_participants) // num_folds
    participant_groups = []
    
    for i in range(num_folds):
        if i == num_folds - 1:
            # Last fold gets remaining participants
            participant_groups.append(all_participants[i * fold_size:])
        else:
            participant_groups.append(all_participants[i * fold_size:(i + 1) * fold_size])
    
    # Create folds: each group takes turn being validation set
    folds = []
    for fold_idx in range(num_folds):
        val_participants = participant_groups[fold_idx]
        
        # Remaining participants for training
        train_participants = []
        for i in range(num_folds):
            if i != fold_idx:
                train_participants.extend(participant_groups[i])
        
        folds.append((train_participants, val_participants))
        
        print(f"Fold {fold_idx}: Train={len(train_participants)} participants, "
              f"Val={len(val_participants)} participants")
    
    return folds


def create_interparticipant_folds_custom_ratio(participants, train_ratio=0.8, val_ratio=0.2,
                                                num_folds=5, seed=42):
    """
    Create inter-participant folds with custom train/val ratios.
    
    Note: For K-fold CV, the val ratio is approximately 1/num_folds.
    The train_ratio and val_ratio are applied within each fold.
    
    Args:
        participants: List of participant IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        num_folds: Number of folds
        seed: Random seed
    
    Returns:
        List of tuples: [(train_participants, val_participants), ...]
    """
    np.random.seed(seed)
    all_participants = list(participants)
    
    np.random.shuffle(all_participants)
    
    # Create folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    folds = []
    participant_array = np.array(all_participants)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(participant_array)):
        train_participants = participant_array[train_idx].tolist()
        val_participants = participant_array[val_idx].tolist()
        
        folds.append((train_participants, val_participants))
    
    return folds


def get_participant_statistics(image_base_path, csv_path, participants):
    """
    Get statistics for a set of participants.
    
    Args:
        image_base_path: Base path to image folders
        csv_path: Path to label CSV
        participants: List of participant IDs
    
    Returns:
        Dictionary with statistics
    """
    label_mapping = load_label_mapping(csv_path)
    
    total_samples = 0
    label_counts = {}
    
    for participant in participants:
        participant_dir = os.path.join(image_base_path, participant)
        
        if not os.path.exists(participant_dir):
            print(f"Warning: Participant directory not found: {participant_dir}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(participant_dir) if f.endswith('.jpg')]
        
        for img_file in image_files:
            q_id, _, _ = parse_image_filename(img_file)
            
            if q_id is None:
                continue
            
            # Look up label from CSV
            key = (participant, q_id)
            if key in label_mapping:
                label = label_mapping[key]
                label_counts[label] = label_counts.get(label, 0) + 1
                total_samples += 1
    
    return {
        'num_participants': len(participants),
        'total_samples': total_samples,
        'samples_per_participant': total_samples / len(participants) if len(participants) > 0 else 0,
        'label_distribution': label_counts
    }


def print_fold_statistics(image_base_path, csv_path, folds):
    """
    Print detailed statistics for each fold.
    
    Args:
        image_base_path: Base path to image folders
        csv_path: Path to label CSV
        folds: List of (train_participants, val_participants)
    """
    for fold_idx, (train_p, val_p) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx} STATISTICS")
        print(f"{'='*50}")
        
        train_stats = get_participant_statistics(image_base_path, csv_path, train_p)
        val_stats = get_participant_statistics(image_base_path, csv_path, val_p)
        
        print(f"\nTRAIN SET:")
        print(f"  Participants: {train_p}")
        print(f"  Total samples: {train_stats['total_samples']}")
        print(f"  Label distribution: {train_stats['label_distribution']}")
        
        print(f"\nVALIDATION SET:")
        print(f"  Participants: {val_p}")
        print(f"  Total samples: {val_stats['total_samples']}")
        print(f"  Label distribution: {val_stats['label_distribution']}")


def validate_image_paths(image_base_path, csv_path, participants=None):
    """
    Validate that image files exist and have corresponding labels in CSV.
    
    Args:
        image_base_path: Base path to image folders
        csv_path: Path to label CSV
        participants: List of participant IDs (None for all in folder)
    
    Returns:
        Tuple of (valid_count, missing_label_count, parse_error_count)
    """
    label_mapping = load_label_mapping(csv_path)
    
    if participants is None:
        participants = [d for d in os.listdir(image_base_path) 
                       if os.path.isdir(os.path.join(image_base_path, d))]
    
    valid_count = 0
    missing_label_count = 0
    parse_error_count = 0
    missing_examples = []
    parse_error_examples = []
    
    for participant in participants:
        participant_dir = os.path.join(image_base_path, participant)
        
        if not os.path.exists(participant_dir):
            print(f"Warning: Participant directory not found: {participant_dir}")
            continue
        
        image_files = [f for f in os.listdir(participant_dir) if f.endswith('.jpg')]
        
        for img_file in image_files:
            q_id, _, _ = parse_image_filename(img_file)
            
            if q_id is None:
                parse_error_count += 1
                if len(parse_error_examples) < 10:
                    parse_error_examples.append(os.path.join(participant, img_file))
                continue
            
            # Check if label exists in CSV
            key = (participant, q_id)
            if key in label_mapping:
                valid_count += 1
            else:
                missing_label_count += 1
                if len(missing_examples) < 10:
                    missing_examples.append(f"{participant}/{q_id}")
    
    return (valid_count, missing_label_count, parse_error_count, 
            missing_examples, parse_error_examples)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create inter-participant folds")
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to label CSV file")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to image folders")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Get participants from folder
    participants = [d for d in os.listdir(args.image_path) 
                   if os.path.isdir(os.path.join(args.image_path, d))]
    participants = sorted(participants)
    
    print(f"Found {len(participants)} participants in {args.image_path}")
    print(f"Participants: {participants}")
    
    # Validate images
    print(f"\nValidating images and labels...")
    valid, missing_labels, parse_errors, missing_ex, parse_ex = validate_image_paths(
        args.image_path, args.csv_path, participants
    )
    print(f"Valid images: {valid}")
    print(f"Missing labels: {missing_labels}")
    print(f"Parse errors: {parse_errors}")
    
    if missing_ex:
        print(f"Sample missing labels: {missing_ex[:10]}")
    if parse_ex:
        print(f"Sample parse errors: {parse_ex[:10]}")
    
    # Create folds
    folds = create_interparticipant_folds(participants, num_folds=args.num_folds, seed=args.seed)
    
    # Print statistics
    print_fold_statistics(args.image_path, args.csv_path, folds)