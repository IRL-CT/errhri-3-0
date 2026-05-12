#!/usr/bin/env python3
"""
Train a single final model on the full trainval set (no CV), then evaluate on the test set.

Outputs written to --output_dir:
  config.json           all hyperparameters
  model.pt              trained weights + optimizer state
  training_history.json per-epoch train loss and accuracy
  test_gt.csv           frame-level ground-truth for eval.py
  test_predictions.csv  window-level predictions for eval.py
  eval_results.json     official ERR@HRI metrics
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve repo paths relative to this file so the script works from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)   # badnet_pytorch, get_metrics (co-located in baseline/)
sys.path.insert(0, _REPO)   # eval.py (repo root)

from badnet_pytorch import (
    set_seed, BadNetDatasetNPY, BadNetDatasetWithAugmentation,
    create_model, parse_image_filename,
)
from eval import compute_metrics, compute_temporal_metrics


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train final model on full trainval data')

    # Identity
    p.add_argument('--run_name',   required=True, help='Sweep run name (used for labelling output)')
    p.add_argument('--output_dir', required=True, help='Directory to write all outputs')
    p.add_argument('--track',      required=True, type=int, choices=[1, 2],
                   help='Challenge track (1=BAD, 2=Bad Idea)')

    # Trainval data — update these to match your local dataset layout
    p.add_argument('--csv_path',        default='<path/to/dataset>/trainval_npy/label_data.csv')
    p.add_argument('--npy_base_path',   default='<path/to/dataset>/trainval_npy')
    p.add_argument('--image_base_path', default='<path/to/dataset>/trainval_frames')

    # Test data
    p.add_argument('--test_csv_path',        default=None,
                   help='Path to test label CSV; generated from filenames if missing')
    p.add_argument('--test_npy_base_path',   default=None)
    p.add_argument('--test_image_base_path', default=None)

    # Model architecture
    p.add_argument('--model_type', default='original',
                   choices=['original', 'simple', 'pretrained_resnet18',
                            'pretrained_resnet34', 'pretrained_resnet50'])
    p.add_argument('--activation',       default='relu', choices=['relu', 'sigmoid'])
    p.add_argument('--kernel_size',      type=int,   default=4)
    p.add_argument('--base_filters',     type=int,   default=16)
    p.add_argument('--freeze_backbone',  action='store_true')
    p.add_argument('--dropout',          type=float, default=0.5)

    # Training
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--learning_rate', type=float, default=0.0001)
    p.add_argument('--num_workers',   type=int,   default=4)

    # Data
    p.add_argument('--use_npy',           action='store_true')
    p.add_argument('--use_weighted_loss', action='store_true')
    p.add_argument('--num_augmentations', type=int, default=0)
    p.add_argument('--cache_images',      action='store_true')

    # Temporal / eval window
    p.add_argument('--window_size',  type=int, default=5)
    p.add_argument('--slide_length', type=int, default=1)
    p.add_argument('--fps',          type=int, default=5,
                   help='Frame rate of extracted data (5 for BAD, 30 for Bad Idea)')

    # Misc
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--no_cuda', action='store_true')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (adapted from train_badnet.py)
# ---------------------------------------------------------------------------

def _majority_vote(preds):
    counts = Counter(preds)
    return 1 if counts[1] >= counts[0] else 0


def build_official_submission(all_preds, all_labels, all_probs,
                               all_participant_ids, all_video_ids,
                               window_size, slide_length):
    """Frame-level arrays → window-level pred_df + video-level gt_video_df."""
    groups = {}
    for i, (pid, vid) in enumerate(zip(all_participant_ids, all_video_ids)):
        groups.setdefault((str(pid), str(vid)), []).append(i)

    pred_records, gt_records = [], []
    for (pid, vid), indices in groups.items():
        indices = np.array(indices)
        frame_preds = all_preds[indices]
        frame_true  = all_labels[indices]
        n_frames    = len(indices)
        true_label  = int(Counter(frame_true.tolist()).most_common(1)[0][0])

        gt_records.append({'participant_id': pid, 'video_id': vid,
                           'y_true': true_label, 'n_frames': n_frames})

        has_probs  = all_probs is not None and len(all_probs) > 0
        frame_probs = all_probs[indices] if has_probs else None

        window_id = 0
        for start in range(0, n_frames - window_size + 1, slide_length):
            end = start + window_size
            if frame_probs is not None:
                avg_probs = np.mean(frame_probs[start:end], axis=0)
                rec = {'participant_id': pid, 'video_id': vid,
                       'window_id': window_id, 'y_pred': int(np.argmax(avg_probs)),
                       'y_prob_0': float(avg_probs[0]), 'y_prob_1': float(avg_probs[1])}
            else:
                rec = {'participant_id': pid, 'video_id': vid,
                       'window_id': window_id, 'y_pred': _majority_vote(frame_preds[start:end])}
            pred_records.append(rec)
            window_id += 1

    return pd.DataFrame(pred_records), pd.DataFrame(gt_records)


def build_gt_csv(dataset, output_path):
    """Save frame-level GT CSV (participant_id, video_id, frame_id, y_true)."""
    records = []
    frame_counters = {}
    for i in range(len(dataset)):
        meta = dataset.get_metadata(i)
        pid, vid = meta['participant_id'], meta['video_id']
        key = (pid, vid)
        frame_counters[key] = frame_counters.get(key, 0) + 1
        records.append({'participant_id': pid, 'video_id': vid,
                        'frame_id': frame_counters[key], 'y_true': meta['label']})
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f'Saved GT CSV → {output_path}  ({len(records)} frames)')


def maybe_generate_label_csv(npy_base_path, participants, csv_path):
    """Generate label_data.csv from NPY filenames if the file doesn't exist."""
    if os.path.isfile(csv_path):
        return
    print(f'[INFO] {csv_path} not found — generating from filenames...')
    records = []
    for participant in participants:
        pdir = os.path.join(npy_base_path, participant)
        for fname in sorted(os.listdir(pdir)):
            if not fname.endswith('.npy'):
                continue
            q_id, label, _ = parse_image_filename(fname.replace('.npy', '.png'))
            if q_id is None:
                q_id, label, _ = parse_image_filename(fname.replace('.npy', '.jpg'))
            if q_id is not None:
                records.append({'participant_id': participant, 'q_id': q_id, 'label': label})
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f'[INFO] Generated {csv_path}  ({len(records)} rows)')


def run_inference(model, loader, device):
    """Run model in eval mode, return frame-level arrays."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_pids, all_vids = [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            if hasattr(loader.dataset, 'get_metadata'):
                start_idx = batch_idx * loader.batch_size
                for i in range(inputs.size(0)):
                    idx = start_idx + i
                    if idx < len(loader.dataset):
                        meta = loader.dataset.get_metadata(idx)
                        all_pids.append(meta['participant_id'])
                        all_vids.append(meta['video_id'])

    return (np.array(all_preds), np.array(all_labels), np.array(all_probs),
            np.array(all_pids), np.array(all_vids))


def compute_eval_results(pred_df, gt_video_df, window_size, slide_length, track=2):
    """Compute all official ERR@HRI metrics and return as a JSON-serialisable dict."""
    has_proba = 'y_prob_1' in pred_df.columns

    win = pred_df.merge(gt_video_df[['participant_id', 'video_id', 'y_true']],
                        on=['participant_id', 'video_id'], how='left')
    y_prob_win = win['y_prob_1'].values if has_proba else None
    m_win = compute_metrics(win['y_true'].values, win['y_pred'].values,
                            y_prob_win, level='window')

    vid_pred = (win.groupby(['participant_id', 'video_id'])['y_pred']
                   .agg(_majority_vote).reset_index()
                   .rename(columns={'y_pred': 'y_pred_vid'}))
    vid = vid_pred.merge(gt_video_df[['participant_id', 'video_id', 'y_true']],
                         on=['participant_id', 'video_id'])

    y_prob_vid = None
    if has_proba:
        # Track 2 primary: max y_prob_1 per video; Track 1: mean (secondary AUC)
        agg_fn = 'max' if track == 2 else 'mean'
        vid_prob = (win.groupby(['participant_id', 'video_id'])['y_prob_1']
                       .agg(agg_fn).reset_index()
                       .rename(columns={'y_prob_1': 'y_prob_1_vid'}))
        vid = vid.merge(vid_prob, on=['participant_id', 'video_id'])
        y_prob_vid = vid['y_prob_1_vid'].values

    m_vid = compute_metrics(vid['y_true'].values, vid['y_pred_vid'].values,
                            y_prob_vid, level='video')
    det_pct, avg_fnr = compute_temporal_metrics(pred_df, gt_video_df, window_size, slide_length)

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return float(v)

    return {
        'f1_macro_vid':       _safe(m_vid['f1_macro']),
        'balanced_acc_vid':   _safe(m_vid['balanced_accuracy']),
        'window': {k: _safe(v) if isinstance(v, float) else int(v)
                   for k, v in m_win.items() if k != 'level'},
        'video':  {k: _safe(v) if isinstance(v, float) else int(v)
                   for k, v in m_vid.items() if k != 'level'},
        'temporal': {'det_pct': _safe(det_pct), 'avg_fnr': _safe(avg_fnr)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve all relative paths from this script's directory
    def resolve(p):
        if p is None or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(_HERE, p))

    args.csv_path             = resolve(args.csv_path)
    args.npy_base_path        = resolve(args.npy_base_path)
    args.image_base_path      = resolve(args.image_base_path)
    args.test_csv_path        = resolve(args.test_csv_path)
    args.test_npy_base_path   = resolve(args.test_npy_base_path)
    args.test_image_base_path = resolve(args.test_image_base_path)
    args.output_dir           = resolve(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Enforce Track 1 window_size constraint: window_size ≤ 2 × fps
    if args.track == 1:
        max_window = 2 * args.fps
        if args.window_size > max_window:
            print(f'[WARN] Track 1: window_size={args.window_size} exceeds 2×fps limit '
                  f'({max_window} frames at {args.fps} fps). Capping to {max_window}.')
            args.window_size = max_window

    config = vars(args).copy()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\n{"="*70}')
    print(f'Run      : {args.run_name}')
    print(f'Output   : {args.output_dir}')
    print(f'Track    : {args.track}')
    print(f'{"="*70}')

    # ── Early exit if already fully evaluated ───────────────────────────────
    eval_path  = os.path.join(args.output_dir, 'eval_results.json')
    model_path = os.path.join(args.output_dir, 'model.pt')
    ckpt_path  = os.path.join(args.output_dir, 'checkpoint.pt')

    if os.path.isfile(eval_path):
        print('[INFO] eval_results.json already exists — nothing to do.')
        return

    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device   : {device}')

    use_cuda_pin = device.type == 'cuda'

    # ── Create model ────────────────────────────────────────────────────────
    model = create_model(
        model_type=args.model_type, num_classes=2,
        base_filters=args.base_filters, kernel_size=args.kernel_size,
        activation=args.activation, freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(f'Parameters    : {sum(p.numel() for p in model.parameters()):,}')

    # ── Training (skipped if model.pt already saved) ────────────────────────
    if os.path.isfile(model_path):
        print(f'[INFO] model.pt found — loading weights, skipping training.')
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        history = ckpt.get('history', {})
    else:
        # Load trainval dataset
        base_path = args.npy_base_path if args.use_npy else args.image_base_path
        all_participants = sorted([d for d in os.listdir(base_path)
                                   if os.path.isdir(os.path.join(base_path, d))])
        print(f'Trainval : {len(all_participants)} participants')

        if args.use_npy:
            train_dataset = BadNetDatasetNPY(
                all_participants, args.npy_base_path, args.csv_path,
                num_augmentations=args.num_augmentations,
                cache_images=args.cache_images,
            )
        else:
            train_dataset = BadNetDatasetWithAugmentation(
                all_participants, args.image_base_path, args.csv_path,
                num_augmentations=args.num_augmentations,
            )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=use_cuda_pin)
        print(f'Train samples : {len(train_dataset)}')

        if args.use_weighted_loss:
            counts = np.bincount(train_dataset.labels, minlength=2)
            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum() * 2
            criterion = nn.CrossEntropyLoss(
                label_smoothing=0.1,
                weight=torch.tensor(weights, dtype=torch.float).to(device),
            )
            print(f'Class weights : {weights}')
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False,
        )

        # Resume from mid-run checkpoint if one exists
        history     = {'train_loss': [], 'train_acc': []}
        start_epoch = 0
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            history     = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            print(f'[INFO] Resuming from checkpoint at epoch {start_epoch}.')

        print(f'\nTraining from epoch {start_epoch + 1} to {args.epochs}...')

        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = correct / total
            scheduler.step(epoch_loss)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print(f'  Epoch {epoch+1:4d}/{args.epochs}: loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}')

            # Save checkpoint every 25 epochs to allow resuming after interruption
            if (epoch + 1) % 25 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'config': config,
                }, ckpt_path)

        # Save final model and remove the rolling checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history,
        }, model_path)
        print(f'\nSaved model → {model_path}')

        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)

    # ── Test-set evaluation ─────────────────────────────────────────────────
    test_base_path = args.test_npy_base_path if args.use_npy else args.test_image_base_path
    if not (test_base_path and os.path.isdir(test_base_path)):
        print(f'\n[WARN] Test base path not found: {test_base_path} — skipping evaluation.')
        return

    test_participants = sorted([d for d in os.listdir(test_base_path)
                                if os.path.isdir(os.path.join(test_base_path, d))])
    if not test_participants:
        print('[WARN] No test participants found — skipping evaluation.')
        return

    # Generate label CSV from filenames if it doesn't exist
    if args.test_csv_path is None:
        args.test_csv_path = os.path.join(args.output_dir, 'test_label_data.csv')
    if args.use_npy and not os.path.isfile(args.test_csv_path):
        generated_csv = os.path.join(args.output_dir, 'test_label_data.csv')
        maybe_generate_label_csv(test_base_path, test_participants, generated_csv)
        args.test_csv_path = generated_csv

    if not os.path.isfile(args.test_csv_path):
        print(f'[WARN] Test CSV not available: {args.test_csv_path} — skipping evaluation.')
        return

    print(f'\nEvaluating on {len(test_participants)} test participants...')
    test_dataset = BadNetDatasetNPY(
        test_participants, test_base_path, args.test_csv_path,
        num_augmentations=0, cache_images=args.cache_images,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_cuda_pin)

    all_preds, all_labels, all_probs, all_pids, all_vids = run_inference(
        model, test_loader, device,
    )

    gt_csv_path = os.path.join(args.output_dir, 'test_gt.csv')
    build_gt_csv(test_dataset, gt_csv_path)

    pred_df, gt_video_df = build_official_submission(
        all_preds, all_labels, all_probs, all_pids, all_vids,
        args.window_size, args.slide_length,
    )
    pred_csv_path = os.path.join(args.output_dir, 'test_predictions.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    print(f'Saved predictions → {pred_csv_path}  ({len(pred_df)} windows)')

    results = compute_eval_results(pred_df, gt_video_df, args.window_size, args.slide_length,
                                   track=args.track)
    results['run_name'] = args.run_name
    results['config']   = config

    eval_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*70}')
    print(f'TEST RESULTS  —  {args.run_name}')
    if args.track == 2:
        auc = results["video"].get("auc")
        print(f'  AUC-ROC   (video, primary): {auc:.4f}' if auc is not None else '  AUC-ROC: N/A (no probabilities)')
    print(f'  F1 macro  (video): {results["f1_macro_vid"]:.4f}')
    print(f'  Bal. acc  (video): {results["balanced_acc_vid"]:.4f}')
    if results["temporal"]["det_pct"] is not None:
        print(f'  Det. time       : {results["temporal"]["det_pct"]:.1f}%')
    if results["temporal"]["avg_fnr"] is not None:
        print(f'  Avg FNR         : {results["temporal"]["avg_fnr"]:.4f}')
    print(f'Saved eval results → {eval_path}')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
