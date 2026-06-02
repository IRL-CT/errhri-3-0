#!/usr/bin/env python3
"""
ERR@HRI 3.0 — Submission Validator / Official Evaluator
========================================================
Dual-mode script that can be used by participants and organizers:

  PARTICIPANT MODE (no --gt flag)
  --------------------------------
  Validates the format of your submission CSV and checks coverage against
  the expected test-set video list. Labels are NOT required. Run this
  before submitting to catch common errors.

  ORGANIZER MODE (with --gt flag)
  --------------------------------
  Adds full metric computation using the frame-level ground-truth CSV.
  All participant-mode checks are still run first.

Usage
-----
  # Participants — format check only
  python validate_submission.py \\
      --pred my_submission_track1.csv \\
      --track 1 --fps 5 --window_size 10 --slide 5

  # Organizers — full evaluation
  python validate_submission.py \\
      --pred submission_track1.csv \\
      --track 1 --fps 5 --window_size 10 --slide 5 \\
      --gt ground_truth_track1.csv [--out per_video_results.csv]

Submission CSV format (one row per window per clip)
----------------------------------------------------
  participant_id   str   must match test set
  video_id         str   must match test set
  window_id        int   0-indexed window position within the clip
  y_pred           int   window prediction: 0 or 1
  y_prob_0         float probability for class 0  (required)
  y_prob_1         float probability for class 1  (required)

Window parameter constraints
-----------------------------
  slide <= window_size              (both tracks)
  window_size <= 2 * fps            (Track 1 only — max 2-second window)

Expected test video lists are bundled alongside this script:
  test_video_list_track1.csv   (326 videos, 9 participants)
  test_video_list_track2.csv   (180 videos, 6 participants)
"""

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

TRACK_META = {
    1: {
        "name": "Track 1 — Bystander Reaction Detection (BAD Dataset)",
        "label_0": "Control",
        "label_1": "Failure",
        "video_list": HERE / "test_video_list_track1.csv",
    },
    2: {
        "name": "Track 2 — Anticipatory Response Prediction (Bad Idea Dataset)",
        "label_0": "Well",
        "label_1": "Poorly",
        "video_list": HERE / "test_video_list_track2.csv",
    },
}

REQUIRED_COLS   = {"participant_id", "video_id", "window_id", "y_pred"}
PROB_COLS       = {"y_prob_0", "y_prob_1"}
PROB_TOLERANCE  = 0.01   # |y_prob_0 + y_prob_1 - 1| must be within this


# ---------------------------------------------------------------------------
# Format validation (participant mode)
# ---------------------------------------------------------------------------

def validate_format(df, has_proba, track, fps, window_size, slide):
    """
    Run all format checks. Prints PASS/FAIL for each check.
    Returns True if all checks pass, False otherwise.
    """
    errors   = []
    warnings_ = []
    sep = "-" * 60

    print(f"\n{sep}")
    print("  FORMAT CHECKS")
    print(sep)

    def check(label, condition, msg=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if not condition:
            errors.append(f"       → {msg}" if msg else f"       → {label}")
        return condition

    # 1. Required columns
    missing_req = REQUIRED_COLS - set(df.columns)
    check("Required columns present",
          len(missing_req) == 0,
          f"Missing columns: {missing_req}")

    # 2. Probability columns
    missing_prob = PROB_COLS - set(df.columns)
    if missing_prob:
        print(f"  [WARN] Optional columns {missing_prob} not found — AUC cannot be computed.")
        warnings_.append(f"Missing probability columns: {missing_prob}")
    else:
        # 3. y_prob values parseable and in [0, 1]
        ok_range = (df["y_prob_0"].between(0, 1, inclusive="both") &
                    df["y_prob_1"].between(0, 1, inclusive="both")).all()
        check("y_prob_0 and y_prob_1 in [0, 1]", ok_range,
              "Some probability values are outside [0, 1].")

        # 4. Probabilities sum to 1
        prob_sum = (df["y_prob_0"] + df["y_prob_1"])
        ok_sum = (prob_sum - 1.0).abs().le(PROB_TOLERANCE).all()
        check(f"y_prob_0 + y_prob_1 ≈ 1 (tol {PROB_TOLERANCE})", ok_sum,
              f"Max deviation from 1.0: {(prob_sum - 1.0).abs().max():.4f}")

    # 5. y_pred binary
    if "y_pred" in df.columns:
        check("y_pred is binary (0 or 1)",
              df["y_pred"].isin([0, 1]).all(),
              f"Unique values found: {sorted(df['y_pred'].unique())}")

    # 6. window_id is integer >= 0
    if "window_id" in df.columns:
        check("window_id is non-negative integer",
              (df["window_id"] >= 0).all() and pd.api.types.is_integer_dtype(df["window_id"]),
              "window_id must be a non-negative integer.")

    # 7. No duplicate (participant_id, video_id, window_id)
    if all(c in df.columns for c in ["participant_id", "video_id", "window_id"]):
        n_dups = df[["participant_id", "video_id", "window_id"]].duplicated().sum()
        check("No duplicate (participant_id, video_id, window_id) rows",
              n_dups == 0,
              f"{n_dups} duplicate row(s) found.")

    # 8. Window parameter constraints
    if track == 1:
        max_ws = 2 * fps
        check(f"window_size ({window_size}) ≤ 2 × fps ({fps}) = {max_ws}  [Track 1]",
              window_size <= max_ws,
              f"window_size={window_size} exceeds the 2-second cap at {fps} fps.")
    check(f"slide ({slide}) ≤ window_size ({window_size})",
          slide <= window_size,
          f"slide={slide} creates gaps between windows (must be ≤ window_size).")

    # 9. window_id starts at 0 and is sequential per clip
    if all(c in df.columns for c in ["participant_id", "video_id", "window_id"]):
        bad_clips = []
        for (pid, vid), grp in df.groupby(["participant_id", "video_id"]):
            ids = sorted(grp["window_id"].tolist())
            expected = list(range(len(ids)))
            if ids != expected:
                bad_clips.append(f"{pid}/{vid}: got {ids[:5]}{'...' if len(ids) > 5 else ''}")
        check("window_id is 0-indexed and sequential per clip",
              len(bad_clips) == 0,
              f"{len(bad_clips)} clip(s) have non-sequential window_ids. "
              f"First offenders: {bad_clips[:3]}")

    if errors:
        print(f"\n  {len(errors)} error(s) found:")
        for e in errors:
            print(e)
        return False
    else:
        print("\n  All format checks passed.")
        return True


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def check_coverage(df, track):
    """Cross-check submission against expected test-set video list."""
    sep = "-" * 60
    print(f"\n{sep}")
    print("  COVERAGE CHECK")
    print(sep)

    video_list_path = TRACK_META[track]["video_list"]
    if not video_list_path.exists():
        print(f"  [WARN] Expected video list not found: {video_list_path}")
        print("         Skipping coverage check.")
        return

    expected = pd.read_csv(video_list_path, dtype=str)
    expected_keys = set(zip(expected["participant_id"], expected["video_id"]))

    sub_keys = set(zip(df["participant_id"].astype(str), df["video_id"].astype(str)))

    missing  = expected_keys - sub_keys
    extra    = sub_keys - expected_keys

    print(f"  Expected videos : {len(expected_keys)}")
    print(f"  Submitted videos: {len(sub_keys)}")

    if missing:
        print(f"\n  [WARN] {len(missing)} video(s) missing from submission "
              f"(will be imputed as all-zero predictions during evaluation):")
        for pid, vid in sorted(missing)[:10]:
            print(f"         participant {pid} — {vid}")
        if len(missing) > 10:
            print(f"         ... and {len(missing) - 10} more")
    else:
        print("  [PASS] All expected videos are present.")

    if extra:
        print(f"\n  [WARN] {len(extra)} video(s) in submission not in expected test set "
              f"(will be ignored during evaluation):")
        for pid, vid in sorted(extra)[:5]:
            print(f"         participant {pid} — {vid}")
        if len(extra) > 5:
            print(f"         ... and {len(extra) - 5} more")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_submission_summary(df, has_proba):
    sep = "-" * 60
    print(f"\n{sep}")
    print("  SUBMISSION SUMMARY")
    print(sep)

    n_clips   = df.groupby(["participant_id", "video_id"]).ngroups
    n_windows = len(df)
    wins_per_clip = df.groupby(["participant_id", "video_id"]).size()

    print(f"  Clips (participant + video pairs): {n_clips}")
    print(f"  Total windows                    : {n_windows}")
    print(f"  Windows per clip — min/mean/max  : "
          f"{wins_per_clip.min()}/{wins_per_clip.mean():.1f}/{wins_per_clip.max()}")

    if "y_pred" in df.columns:
        pred_counts = df["y_pred"].value_counts().sort_index()
        for label, count in pred_counts.items():
            print(f"  y_pred={label} (window level)         : {count} ({count/n_windows*100:.1f}%)")

    if has_proba:
        print(f"  y_prob_1 — min/mean/max          : "
              f"{df['y_prob_1'].min():.3f}/"
              f"{df['y_prob_1'].mean():.3f}/"
              f"{df['y_prob_1'].max():.3f}")


# ---------------------------------------------------------------------------
# Full evaluation (organizer mode) — ported from eval.py
# ---------------------------------------------------------------------------

def _assert_constant_label(series):
    unique = series.unique()
    if len(unique) != 1:
        raise ValueError(
            f"Inconsistent y_true labels within a clip: {sorted(unique)}. "
            "All frames in a clip must share the same label."
        )
    return int(unique[0])


def load_ground_truth(path):
    df = pd.read_csv(path)
    missing = {"participant_id", "video_id", "frame_id", "y_true"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Ground-truth CSV missing columns: {missing}")
    df["participant_id"] = df["participant_id"].astype(str)
    df["video_id"]       = df["video_id"].astype(str)
    df["frame_id"]       = df["frame_id"].astype(int)
    if not df["y_true"].isin([0, 1]).all():
        sys.exit("[ERROR] y_true must be binary (0 or 1).")
    gt_video = (df.groupby(["participant_id", "video_id"])["y_true"]
                  .agg(lambda x: _assert_constant_label(x))
                  .reset_index())
    frame_counts = (df.groupby(["participant_id", "video_id"])["frame_id"]
                      .count().reset_index()
                      .rename(columns={"frame_id": "n_frames"}))
    return gt_video.merge(frame_counts, on=["participant_id", "video_id"])


def compute_metrics(y_true, y_pred, y_prob_pos=None, level=""):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_pos   = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    f1_neg   = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    prec     = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec      = recall_score(y_true,    y_pred, average="binary", zero_division=0)
    acc      = accuracy_score(y_true,  y_pred)
    bal      = balanced_accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    auc = None
    if y_prob_pos is not None:
        try:
            auc = roc_auc_score(y_true, y_prob_pos)
        except Exception as e:
            warnings.warn(f"AUC ({level}): {e}")
    return dict(level=level, n=len(y_true),
                n_pos=int((y_true == 1).sum()), n_neg=int((y_true == 0).sum()),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1_macro=f1_macro, f1_pos=f1_pos, f1_neg=f1_neg,
                precision=prec, recall=rec, accuracy=acc,
                balanced_accuracy=bal, auc=auc)


def compute_temporal_metrics(pred_df, gt_video, window_size, slide):
    detection_pcts = []
    fnrs           = []
    for (pid, vid), win_grp in pred_df.groupby(["participant_id", "video_id"]):
        gt_row = gt_video[(gt_video["participant_id"] == pid) &
                          (gt_video["video_id"] == vid)]
        if gt_row.empty:
            continue
        true_label = int(gt_row["y_true"].iloc[0])
        n_frames   = int(gt_row["n_frames"].iloc[0])
        if true_label != 1:
            continue
        win_grp        = win_grp.sort_values("window_id")
        preds          = win_grp["y_pred"].values
        expected_n_win = max((n_frames - window_size) // slide + 1, 1)
        n_missed       = int((preds != 1).sum())
        fnrs.append(n_missed / expected_n_win)
        first_correct = next((i for i, p in enumerate(preds) if p == true_label), None)
        counts        = Counter(preds)
        video_pred    = 1 if counts[1] >= counts[0] else 0
        if first_correct is not None and video_pred == true_label:
            detected_at = first_correct * slide + window_size
            detection_pcts.append(detected_at / n_frames * 100.0)
    avg_det = float(np.mean(detection_pcts)) if detection_pcts else float("nan")
    avg_fnr = float(np.mean(fnrs))           if fnrs           else float("nan")
    return avg_det, avg_fnr


def print_block(m, label_0, label_1, track=1):
    if track == 1:
        f1_star, f1_tag   = "  ★", " [CO-PRIMARY]"
        bal_star, bal_tag = "  ★", " [CO-PRIMARY]"
        auc_star, auc_tag = "   ", ""
    else:
        f1_star, f1_tag   = "   ", ""
        bal_star, bal_tag = "   ", ""
        auc_star, auc_tag = "  ★", " [PRIMARY]"

    print(f"  n={m['n']}  ({label_1}: {m['n_pos']}, {label_0}: {m['n_neg']})")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"             {label_0:>10}  {label_1:>10}")
    print(f"  {label_0:>10}  {m['tn']:>10}  {m['fp']:>10}")
    print(f"  {label_1:>10}  {m['fn']:>10}  {m['tp']:>10}")
    print(f"{f1_star} F1 macro{f1_tag}         : {m['f1_macro']:.4f}")
    print(f"{bal_star} Balanced Accuracy{bal_tag}: {m['balanced_accuracy']:.4f}")
    print(f"   F1 ({label_1:>8})              : {m['f1_pos']:.4f}")
    print(f"   F1 ({label_0:>8})              : {m['f1_neg']:.4f}")
    print(f"   Precision  ({label_1:>8})       : {m['precision']:.4f}")
    print(f"   Recall     ({label_1:>8})       : {m['recall']:.4f}")
    print(f"   Accuracy                       : {m['accuracy']:.4f}")
    if m["auc"] is not None:
        print(f"{auc_star} AUC-ROC{auc_tag}                    : {m['auc']:.4f}")


def run_full_evaluation(pred_df, has_proba, gt_path, track, fps, window_size, slide, out_path):
    gt_video = load_ground_truth(gt_path)

    label_0 = TRACK_META[track]["label_0"]
    label_1 = TRACK_META[track]["label_1"]

    # Impute missing predictions
    gt_keys   = set(zip(gt_video.participant_id, gt_video.video_id))
    pred_keys = set(zip(pred_df.participant_id,  pred_df.video_id))
    missing_vids = gt_keys - pred_keys
    if missing_vids:
        print(f"[WARN] {len(missing_vids)} video(s) missing — imputing all-zero prediction.")
        imputed = []
        for pid, vid in missing_vids:
            row = {"participant_id": str(pid), "video_id": str(vid),
                   "window_id": 0, "y_pred": 0}
            if has_proba:
                row["y_prob_0"] = 1.0
                row["y_prob_1"] = 0.0
            imputed.append(row)
        pred_df = pd.concat([pred_df, pd.DataFrame(imputed)], ignore_index=True)

    extra_vids = pred_keys - gt_keys
    if extra_vids:
        print(f"[WARN] {len(extra_vids)} submitted video(s) not in GT — ignored.")
        valid_mask = pd.MultiIndex.from_arrays(
            [pred_df.participant_id, pred_df.video_id]).isin(gt_keys)
        pred_df = pred_df[valid_mask]

    # Window-level metrics
    win = pred_df.merge(gt_video[["participant_id", "video_id", "y_true"]],
                        on=["participant_id", "video_id"], how="left")
    y_prob_win = win["y_prob_1"].values if has_proba else None
    m_win = compute_metrics(win["y_true"].values, win["y_pred"].values,
                            y_prob_win, level="window")

    # Video-level majority vote
    def _majority_vote(x):
        counts = Counter(x)
        return 1 if counts[1] >= counts[0] else 0

    vid_pred = (win.groupby(["participant_id", "video_id"])["y_pred"]
                   .agg(_majority_vote)
                   .reset_index().rename(columns={"y_pred": "y_pred_vid"}))
    vid = vid_pred.merge(gt_video[["participant_id", "video_id", "y_true"]],
                         on=["participant_id", "video_id"])

    y_prob_vid = None
    if has_proba:
        agg_fn = "max" if track == 2 else "mean"
        vid_prob = (win.groupby(["participant_id", "video_id"])["y_prob_1"]
                       .agg(agg_fn).reset_index()
                       .rename(columns={"y_prob_1": "y_prob_1_vid"}))
        vid = vid.merge(vid_prob, on=["participant_id", "video_id"])
        y_prob_vid = vid["y_prob_1_vid"].values

    m_vid = compute_metrics(vid["y_true"].values, vid["y_pred_vid"].values,
                            y_prob_vid, level="video")

    det_pct, avg_fnr = compute_temporal_metrics(pred_df, gt_video, window_size, slide)

    # Print report
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ERR@HRI 3.0 — {TRACK_META[track]['name']}")
    print(f"  Window size={window_size}, slide={slide}, fps={fps}")
    print(sep)

    print(f"\n── WINDOW LEVEL  (size={window_size}, slide={slide}) ──────────────────")
    print_block(m_win, label_0, label_1, track=track)

    if track == 2:
        print(f"\n── VIDEO LEVEL  (majority vote for F1; max y_prob_1 for AUC) ──")
    else:
        print(f"\n── VIDEO LEVEL  (majority vote across windows) ────────────")
    print_block(m_vid, label_0, label_1, track=track)

    print(f"\n── TEMPORAL METRICS  (positive-class videos) ──────────────")
    if not np.isnan(det_pct):
        print(f"   Earliest Detection Time  : {det_pct:.1f}%  (lower is better)")
    else:
        print(f"   Earliest Detection Time  : N/A")
    if not np.isnan(avg_fnr):
        print(f"   Avg FNR per {label_1} video : {avg_fnr:.4f}")
    else:
        print(f"   Avg FNR per {label_1} video : N/A")
    print()

    if out_path:
        vid["correct"] = (vid["y_true"] == vid["y_pred_vid"]).astype(int)
        vid.to_csv(out_path, index=False)
        print(f"[INFO] Per-video results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ERR@HRI 3.0 Submission Validator / Official Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pred",        required=True,           help="Submission CSV (window-level predictions)")
    parser.add_argument("--track",       required=True, type=int, choices=[1, 2], help="Challenge track (1 or 2)")
    parser.add_argument("--fps",         required=True, type=int, help="Frame rate used for feature extraction")
    parser.add_argument("--window_size", required=True, type=int, help="Number of frames per window")
    parser.add_argument("--slide",       required=True, type=int, help="Step size between windows (frames)")
    parser.add_argument("--gt",          default=None,            help="[Organizer] Ground-truth CSV (frame-level). Enables full evaluation.")
    parser.add_argument("--out",         default=None,            help="[Organizer] Save per-video results to this CSV path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ERR@HRI 3.0 Submission Validator")
    print(f"  {TRACK_META[args.track]['name']}")
    print(f"  fps={args.fps}  window_size={args.window_size}  slide={args.slide}")
    if args.gt:
        print(f"  Mode: FULL EVALUATION (organizer)")
    else:
        print(f"  Mode: FORMAT CHECK ONLY (participant)")
    print(f"{'='*60}")

    # Load submission
    try:
        df = pd.read_csv(args.pred)
    except Exception as e:
        sys.exit(f"[ERROR] Cannot read submission file: {e}")

    # Cast types
    for col in ["participant_id", "video_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "window_id" in df.columns:
        df["window_id"] = pd.to_numeric(df["window_id"], errors="coerce").astype("Int64")
    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    has_proba = PROB_COLS.issubset(df.columns)
    if has_proba:
        df["y_prob_0"] = pd.to_numeric(df["y_prob_0"], errors="coerce")
        df["y_prob_1"] = pd.to_numeric(df["y_prob_1"], errors="coerce")
        if df[["y_prob_0", "y_prob_1"]].isna().any().any():
            print("[WARN] Some probability values could not be parsed as numbers — "
                  "treating as missing.")
            has_proba = False

    # Format checks (always)
    format_ok = validate_format(df, has_proba, args.track, args.fps, args.window_size, args.slide)

    # Coverage check (always, against bundled expected video list)
    check_coverage(df, args.track)

    # Submission summary
    print_submission_summary(df, has_proba)

    if not format_ok:
        print(f"\n{'='*60}")
        print("  RESULT: SUBMISSION IS INVALID — please fix the errors above.")
        print(f"{'='*60}\n")
        sys.exit(1)

    if args.gt is None:
        print(f"\n{'='*60}")
        print("  RESULT: FORMAT OK")
        print("  Your submission passed all format checks.")
        print("  To see full metrics, re-run with --gt <ground_truth.csv>")
        print("  (organizer use only — ground truth is not released to participants).")
        print(f"{'='*60}\n")
    else:
        # Full evaluation
        # Drop rows with NaN in critical columns before eval
        df_eval = df.dropna(subset=["participant_id", "video_id", "window_id", "y_pred"])
        df_eval = df_eval.copy()
        df_eval["y_pred"] = df_eval["y_pred"].astype(int)
        df_eval["window_id"] = df_eval["window_id"].astype(int)
        run_full_evaluation(df_eval, has_proba, args.gt, args.track,
                            args.fps, args.window_size, args.slide, args.out)


if __name__ == "__main__":
    main()
