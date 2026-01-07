#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet PyTorch - Training Script with NPY support

Supports both JPG and NPY datasets for training.
Train/Val only - no test set.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import wandb

from badnet_pytorch import (
    set_seed, BadNetDatasetWithAugmentation, BadNetDatasetNPY,
    BadNetPretrained, BadNetSimple, create_model, BadNetCNN,
    create_interparticipant_folds, train_fold
)
from get_metrics import get_test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train BADNet PyTorch model")
    
    # Data paths
    parser.add_argument("--csv_path", type=str, 
                        default="../../data_badidea/trainval/label_data.csv",
                        help="Path to CSV file with labels")
    parser.add_argument("--image_base_path", type=str, 
                        default="../../data_badidea/trainval",
                        help="Base path to trainval image folders")
    parser.add_argument("--npy_base_path", type=str,
                        default="../../data_badidea/trainval_npy",
                        help="Base path to trainval NPY folders")

    # Model hyperparameters
    parser.add_argument("--activation", type=str, default="relu", 
                        choices=["relu", "sigmoid"], help="Activation function")
    parser.add_argument("--kernel_size", type=int, default=4, help="Kernel size for conv layers")
    parser.add_argument("--base_filters", type=int, default=16, help="Base number of filters")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="original",
                        choices=["original", "simple", "pretrained_resnet18", "pretrained_resnet34", "pretrained_resnet50"],
                        help="Model architecture type")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze pretrained backbone")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for pretrained models")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Cross-validation
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--fold", type=int, default=None, 
                        help="Specific fold to train (None for all folds)")
    
    # Data format
    parser.add_argument("--use_npy", action="store_true", help="Use NPY files instead of JPG")
    parser.add_argument("--use_weighted_loss", default=False, action="store_true", help="Use weighted loss function")
    parser.add_argument("--num_augmentations", type=int, default=0, help="Number of augmentations per image")
    parser.add_argument("--cache_images", action="store_true", help="Cache images in memory for faster loading")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--wandb_project", type=str, default="errhri_badidea_baseline", 
                        help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    return parser.parse_args()


def evaluate_model(model, dataloader, device, num_classes=2):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Get metrics using get_test_metrics (with tolerance)
    metrics = get_test_metrics(all_preds, all_labels, tolerance=1)
    
    # Add kappa and confusion matrix
    kappa = cohen_kappa_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics['kappa'] = kappa
    metrics['confusion_matrix'] = conf_matrix
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1 Score:  {metrics['test_f1']:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\nTolerant Metrics (tolerance=1):")
    print(f"Accuracy:  {metrics['test_accuracy_tolerant']:.4f}")
    print(f"Precision: {metrics['test_precision_tolerant']:.4f}")
    print(f"Recall:    {metrics['test_recall_tolerant']:.4f}")
    print(f"F1 Score:  {metrics['test_f1_tolerant']:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Device: {device}")

    # Sweep configuration
    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "avg_val_accuracy"},
        "parameters": {
            "activation": {"values": ['relu', 'sigmoid']},
            "kernel_size": {"values": [2, 4, 6, 8]},
            "base_filters": {"values": [16, 32, 64]},
            "learning_rate": {"values": [0.0001, 0.001, 0.00001]},
            "batch_size": {"values": [16, 32, 64]},
            "seed": {"values": [42, 1369]},
            "epochs": {"values": [100]},
            "num_folds": {"values": [5]},
            "csv_path": {"values": [args.csv_path]},
            "npy_base_path": {"values": [args.npy_base_path]},
            "image_base_path": {"values": [args.image_base_path]},
            "patience": {"values": [args.patience]},
            "num_workers": {"values": [args.num_workers]},
            "num_augmentations": {"values": [0,2,3]},
            "model_type": {"values": ['original', 'simple', 'pretrained_resnet18', 'pretrained_resnet34']},
            "freeze_backbone": {"values": [True, False]},
            "dropout": {"values": [0.3, 0.5, 0.7]},
            "use_npy": {"values": [True]},
            "cache_images": {"values": [True]},
            "use_weighted_loss": {"values": [True, False]},
        },
    }
    
    def train_wrapper():
        wandb.init()
        config = wandb.config
        
        # Update args with sweep config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        set_seed(args.seed)
        
        # Get participants from trainval folder
        image_path = args.npy_base_path if args.use_npy else args.image_base_path
        all_participants = [d for d in os.listdir(image_path) 
                           if os.path.isdir(os.path.join(image_path, d))]
        all_participants = sorted(all_participants)
        
        print(f"Found {len(all_participants)} participants in {image_path}")
        print(f"Participants: {all_participants}")
        
        # Number of classes is always 2 (binary classification)
        num_classes = 2
        print(f"Number of classes: {num_classes}")
        
        # Create folds
        print(f"\nCreating {args.num_folds} inter-participant folds...")
        folds = create_interparticipant_folds(all_participants, num_folds=args.num_folds, seed=args.seed)
        
        # Determine which folds to train
        if args.fold is not None:
            folds_to_train = [args.fold]
        else:
            folds_to_train = range(len(folds))
        
        all_val_metrics = []
        
        for fold_idx in folds_to_train:
            train_participants, val_participants = folds[fold_idx]
            
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx}")
            print(f"{'='*60}")
            print(f"Train participants: {train_participants}")
            print(f"Val participants: {val_participants}")
            
            # Create datasets
            if args.use_npy:
                train_dataset = BadNetDatasetNPY(
                    train_participants, args.npy_base_path, args.csv_path,
                    num_augmentations=args.num_augmentations,
                    cache_images=args.cache_images
                )
                val_dataset = BadNetDatasetNPY(
                    val_participants, args.npy_base_path, args.csv_path,
                    num_augmentations=0,
                    cache_images=args.cache_images
                )
            else:
                train_dataset = BadNetDatasetWithAugmentation(
                    train_participants, args.image_base_path, args.csv_path,
                    num_augmentations=args.num_augmentations
                )
                val_dataset = BadNetDatasetWithAugmentation(
                    val_participants, args.image_base_path, args.csv_path,
                    num_augmentations=0
                )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=use_cuda)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=use_cuda)
            
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            
            # Create model
            print("\nCreating model...")
            model = create_model(
                model_type=args.model_type,
                num_classes=num_classes,
                base_filters=args.base_filters,
                kernel_size=args.kernel_size,
                activation=args.activation,
                freeze_backbone=args.freeze_backbone,
                dropout=args.dropout
            )
            model = model.to(device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Loss and optimizer
            if args.use_weighted_loss:
                # Compute class weights
                label_counts = np.bincount(train_dataset.labels, minlength=num_classes)
                class_weights = 1.0 / (label_counts + 1e-6)
                class_weights = class_weights / class_weights.sum() * num_classes
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
                print(f"Using weighted loss with class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights_tensor if args.use_weighted_loss else None)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Train
            checkpoint_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}')
            print(f"\nStarting training for {args.epochs} epochs...")
            print(f"Checkpoints will be saved to {checkpoint_dir}")
            
            history = train_fold(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, args.epochs, patience=args.patience, checkpoint_dir=checkpoint_dir
            )            
            # Log training history to W&B
            for epoch_idx in range(len(history['train_loss'])):
                wandb.log({
                    f'fold_{fold_idx}_train_loss': history['train_loss'][epoch_idx],
                    f'fold_{fold_idx}_train_acc': history['train_acc'][epoch_idx],
                    f'fold_{fold_idx}_val_loss': history['val_loss'][epoch_idx],
                    f'fold_{fold_idx}_val_acc': history['val_acc'][epoch_idx],
                    'epoch': epoch_idx + 1,
                    'fold': fold_idx
                })
                        
            # Evaluate on validation set
            print(f"\nFinal evaluation on validation set...")
            metrics = evaluate_model(model, val_loader, device, num_classes=num_classes)
            all_val_metrics.append(metrics)
            
            # Log validation metrics to W&B
            wandb.log({
                f'fold_{fold_idx}_val_accuracy': metrics['test_accuracy'],
                f'fold_{fold_idx}_val_precision': metrics['test_precision'],
                f'fold_{fold_idx}_val_recall': metrics['test_recall'],
                f'fold_{fold_idx}_val_f1': metrics['test_f1'],
                f'fold_{fold_idx}_val_accuracy_tolerant': metrics['test_accuracy_tolerant'],
                f'fold_{fold_idx}_val_precision_tolerant': metrics['test_precision_tolerant'],
                f'fold_{fold_idx}_val_recall_tolerant': metrics['test_recall_tolerant'],
                f'fold_{fold_idx}_val_f1_tolerant': metrics['test_f1_tolerant'],
                f'fold_{fold_idx}_val_kappa': metrics['kappa']
            })
            
            # Log prediction probabilities
            probs_df = pd.DataFrame(metrics['probabilities'])
            table = wandb.Table(dataframe=probs_df)
            wandb.log({f"fold_{fold_idx}_prediction_probabilities_table": table})
            
            # Save final model
            final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fold': fold_idx,
                'metrics': metrics,
                'args': vars(args)
            }, final_model_path)
            print(f"Saved final model to {final_model_path}")
        
        # Summary
        if len(all_val_metrics) > 0:
            print(f"\n{'='*60}")
            print("SUMMARY ACROSS ALL FOLDS")
            print(f"{'='*60}")
            
            avg_accuracy = np.mean([m['test_accuracy'] for m in all_val_metrics])
            avg_precision = np.mean([m['test_precision'] for m in all_val_metrics])
            avg_recall = np.mean([m['test_recall'] for m in all_val_metrics])
            avg_f1 = np.mean([m['test_f1'] for m in all_val_metrics])
            avg_kappa = np.mean([m['kappa'] for m in all_val_metrics])
            
            avg_accuracy_tol = np.mean([m['test_accuracy_tolerant'] for m in all_val_metrics])
            avg_precision_tol = np.mean([m['test_precision_tolerant'] for m in all_val_metrics])
            avg_recall_tol = np.mean([m['test_recall_tolerant'] for m in all_val_metrics])
            avg_f1_tol = np.mean([m['test_f1_tolerant'] for m in all_val_metrics])
            
            std_accuracy = np.std([m['test_accuracy'] for m in all_val_metrics])
            std_f1 = np.std([m['test_f1'] for m in all_val_metrics])
            std_accuracy_tol = np.std([m['test_accuracy_tolerant'] for m in all_val_metrics])
            std_f1_tol = np.std([m['test_f1_tolerant'] for m in all_val_metrics])
            
            print(f"Validation Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"Validation Precision: {avg_precision:.4f}")
            print(f"Validation Recall:    {avg_recall:.4f}")
            print(f"Validation F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"Validation Kappa:     {avg_kappa:.4f}")
            print(f"\nTolerant Metrics:")
            print(f"Accuracy (tol):  {avg_accuracy_tol:.4f} ± {std_accuracy_tol:.4f}")
            print(f"Precision (tol): {avg_precision_tol:.4f}")
            print(f"Recall (tol):    {avg_recall_tol:.4f}")
            print(f"F1 Score (tol):  {avg_f1_tol:.4f} ± {std_f1_tol:.4f}")
            
            # Log average metrics to W&B
            wandb.log({
                'avg_val_accuracy': avg_accuracy,
                'avg_val_precision': avg_precision,
                'avg_val_recall': avg_recall,
                'avg_val_f1': avg_f1,
                'avg_val_kappa': avg_kappa,
                'avg_val_accuracy_tolerant': avg_accuracy_tol,
                'avg_val_precision_tolerant': avg_precision_tol,
                'avg_val_recall_tolerant': avg_recall_tol,
                'avg_val_f1_tolerant': avg_f1_tol,
                'std_val_accuracy': std_accuracy,
                'std_val_f1': std_f1,
                'std_val_accuracy_tolerant': std_accuracy_tol,
                'std_val_f1_tolerant': std_f1_tol
            })
            
            wandb.run.summary.update({
                'avg_val_accuracy': avg_accuracy,
                'avg_val_precision': avg_precision,
                'avg_val_recall': avg_recall,
                'avg_val_f1': avg_f1,
                'avg_val_kappa': avg_kappa,
                'avg_val_accuracy_tolerant': avg_accuracy_tol,
                'avg_val_precision_tolerant': avg_precision_tol,
                'avg_val_recall_tolerant': avg_recall_tol,
                'avg_val_f1_tolerant': avg_f1_tol
            })
        
        wandb.finish()
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    print(f"Sweep ID: {sweep_id}")
    print("Starting sweep agent...")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == "__main__":
    main()