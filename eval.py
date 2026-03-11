"""
ERR@HRI 3.0 — Official Evaluation Script
==========================================
Participants submit WINDOW-LEVEL predictions (one row per window per clip)
and declare the window_size and slide they used. The organizer aggregates
the frame-level ground truth using the same parameters — since all frames
within a clip share the same label, the aggregated GT label is always
identical to the video label.

Metrics are reported at two levels:
  1. Window level  — one prediction per window
  2. Video level   — majority vote across windows per (participant, video)

Co-primary ranking metrics (video level):
  1. Macro F1           — equally weights both classes
  2. Balanced Accuracy  — mean recall per class; robust to imbalance

Additional temporal metrics (video level):
  - Earliest Detection Time : % of video elapsed at first correct window
                              (lower is better; reported for positive videos only)
  - FNR per video           : fraction of windows that miss the positive label,
                              averaged over positive-class videos

Label conventions
-----------------
  Track 1 (BAD):       0 = Control,  1 = Failure
  Track 2 (Bad Idea):  0 = Well,     1 = Poorly

Submission CSV columns (required):
  participant_id, video_id, window_id, y_pred
Optional:
  y_prob_0, y_prob_1   (enables AUC-ROC)

Ground-truth CSV columns:
  participant_id, video_id, frame_id, y_true

Usage:
  python eval.py --gt gt.csv --pred sub.csv --track 1 --window_size 5 --slide 1
  python eval.py --gt gt.csv --pred sub.csv --track 2 --window_size 10 --slide 2 --out out.csv
"""

import argparse, sys, warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

TRACK_META = {
    1: {"name": "Track 1 — Bystander Reaction Detection (BAD Dataset)",
        "label_0": "Control", "label_1": "Failure"},
    2: {"name": "Track 2 — Anticipatory Response Prediction (Bad Idea Dataset)",
        "label_0": "Well", "label_1": "Poorly"},
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_ground_truth(path):
    """
    Load frame-level GT. Since all frames in a clip share the same label,
    we immediately collapse to one row per (participant_id, video_id).
    """
    df = pd.read_csv(path)
    missing = {"participant_id", "video_id", "frame_id", "y_true"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Ground-truth missing columns: {missing}")
    df["participant_id"] = df["participant_id"].astype(str)
    df["video_id"]       = df["video_id"].astype(str)
    df["frame_id"]       = df["frame_id"].astype(int)
    if not df["y_true"].isin([0, 1]).all():
        sys.exit("[ERROR] y_true must be binary (0 or 1).")

    # Collapse to video-level GT (label is constant within each clip)
    gt_video = (df.groupby(["participant_id", "video_id"])["y_true"]
                  .agg(lambda x: int(Counter(x).most_common(1)[0][0]))
                  .reset_index())

    # Also keep frame counts per clip for computing n_windows and detection %
    frame_counts = (df.groupby(["participant_id", "video_id"])["frame_id"]
                      .count().reset_index()
                      .rename(columns={"frame_id": "n_frames"}))
    return gt_video.merge(frame_counts, on=["participant_id", "video_id"])


def load_predictions(path):
    """Load window-level predictions submitted by participant."""
    df = pd.read_csv(path)
    missing = {"participant_id", "video_id", "window_id", "y_pred"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Submission missing columns: {missing}")
    df["participant_id"] = df["participant_id"].astype(str)
    df["video_id"]       = df["video_id"].astype(str)
    df["window_id"]      = df["window_id"].astype(int)
    if not df["y_pred"].isin([0, 1]).all():
        sys.exit("[ERROR] y_pred must be binary (0 or 1).")
    if df[["participant_id", "video_id", "window_id"]].duplicated().any():
        sys.exit("[ERROR] Submission has duplicate (participant_id, video_id, window_id) rows.")
    has_proba = {"y_prob_0", "y_prob_1"}.issubset(df.columns)
    if has_proba:
        df["y_prob_0"] = pd.to_numeric(df["y_prob_0"], errors="coerce")
        df["y_prob_1"] = pd.to_numeric(df["y_prob_1"], errors="coerce")
        if df[["y_prob_0", "y_prob_1"]].isna().any().any():
            print("[WARN] Some probability values unparseable — skipping AUC.")
            has_proba = False
    keep = ["participant_id", "video_id", "window_id", "y_pred"] + \
           (["y_prob_0", "y_prob_1"] if has_proba else [])
    return df[keep], has_proba


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob_pos=None, level=""):
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
                n_pos=int((y_true==1).sum()), n_neg=int((y_true==0).sum()),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1_macro=f1_macro, f1_pos=f1_pos, f1_neg=f1_neg,
                precision=prec, recall=rec, accuracy=acc,
                balanced_accuracy=bal, auc=auc)


def compute_temporal_metrics(pred_df, gt_video, window_size, slide):
    """
    For each (participant, video):
      - Earliest Detection Time: % of clip elapsed at the first correct window,
        averaged over correctly classified positive-class clips.
      - FNR per video: fraction of windows that miss the positive label,
        averaged over all positive-class clips.
    """
    detection_pcts = []   # for correctly classified positive clips
    fnrs           = []   # for all positive clips

    for (pid, vid), win_grp in pred_df.groupby(["participant_id", "video_id"]):
        gt_row = gt_video[(gt_video["participant_id"] == pid) &
                          (gt_video["video_id"] == vid)]
        if gt_row.empty:
            continue
        true_label = int(gt_row["y_true"].iloc[0])
        n_frames   = int(gt_row["n_frames"].iloc[0])
        if true_label != 1:
            continue  # temporal metrics only for positive-class clips

        win_grp  = win_grp.sort_values("window_id")
        preds    = win_grp["y_pred"].values
        n_win    = len(preds)

        # FNR: fraction of windows that predict 0 on a positive clip
        fnr = float((preds != 1).sum()) / n_win if n_win > 0 else float("nan")
        fnrs.append(fnr)

        # Earliest detection: first window index where prediction == 1
        first_correct = next((i for i, p in enumerate(preds) if p == true_label), None)
        video_pred = int(Counter(preds).most_common(1)[0][0])
        if first_correct is not None and video_pred == true_label:
            # Frame index of the end of that first correct window
            detected_at = first_correct * slide + window_size
            detection_pcts.append(detected_at / n_frames * 100.0)

    avg_detection_pct = float(np.mean(detection_pcts)) if detection_pcts else float("nan")
    avg_fnr           = float(np.mean(fnrs))           if fnrs           else float("nan")
    return avg_detection_pct, avg_fnr


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_block(m, label_0, label_1, primary=False):
    star = "  ★" if primary else "   "
    tag  = " [CO-PRIMARY]" if primary else ""
    print(f"  n={m['n']}  ({label_1}: {m['n_pos']}, {label_0}: {m['n_neg']})")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"             {label_0:>10}  {label_1:>10}")
    print(f"  {label_0:>10}  {m['tn']:>10}  {m['fp']:>10}")
    print(f"  {label_1:>10}  {m['fn']:>10}  {m['tp']:>10}")
    print(f"{star} F1 macro{tag}         : {m['f1_macro']:.4f}")
    print(f"{star} Balanced Accuracy{tag}: {m['balanced_accuracy']:.4f}")
    print(f"   F1 ({label_1:>8})              : {m['f1_pos']:.4f}")
    print(f"   F1 ({label_0:>8})              : {m['f1_neg']:.4f}")
    print(f"   Precision  ({label_1:>8})       : {m['precision']:.4f}")
    print(f"   Recall     ({label_1:>8})       : {m['recall']:.4f}")
    print(f"   Accuracy                       : {m['accuracy']:.4f}")
    if m["auc"] is not None:
        print(f"   AUC-ROC                        : {m['auc']:.4f}")


def print_report(track, ws, sl, m_win, m_vid, det_pct, avg_fnr, label_0, label_1):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ERR@HRI 3.0 — {TRACK_META[track]['name']}")
    print(f"  Window size={ws}, slide={sl}")
    print(sep)

    print(f"\n── WINDOW LEVEL  (size={ws}, slide={sl}) ──────────────────")
    print_block(m_win, label_0, label_1)

    print(f"\n── VIDEO LEVEL  (majority vote across windows) ────────────")
    print_block(m_vid, label_0, label_1, primary=True)

    print(f"\n── TEMPORAL METRICS  (positive-class videos) ──────────────")
    if not np.isnan(det_pct):
        print(f"   Earliest Detection Time  : {det_pct:.1f}%  (lower is better)")
        print(f"   (avg % of clip elapsed at first correct window,")
        print(f"    over correctly classified {label_1} videos)")
    else:
        print(f"   Earliest Detection Time  : N/A")
    if not np.isnan(avg_fnr):
        print(f"   Avg FNR per {label_1} video : {avg_fnr:.4f}")
        print(f"   (avg fraction of windows missing {label_1} label)")
    else:
        print(f"   Avg FNR per {label_1} video : N/A")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ERR@HRI 3.0 Official Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gt",          required=True,           help="Ground-truth CSV (frame-level)")
    parser.add_argument("--pred",        required=True,           help="Submission CSV (window-level)")
    parser.add_argument("--track",       required=True, type=int, choices=[1, 2])
    parser.add_argument("--window_size", required=True, type=int, help="Declared by participant")
    parser.add_argument("--slide",       required=True, type=int, help="Declared by participant")
    parser.add_argument("--out",         default=None,            help="Save per-video results CSV")
    args = parser.parse_args()

    gt_video           = load_ground_truth(args.gt)   # collapsed to video-level
    pred_df, has_proba = load_predictions(args.pred)
    ws, sl             = args.window_size, args.slide

    if not has_proba:
        print("[INFO] No y_prob_0/y_prob_1 columns — AUC will not be computed.")

    # Check coverage
    gt_keys   = set(zip(gt_video.participant_id, gt_video.video_id))
    pred_keys = set(zip(pred_df.participant_id,  pred_df.video_id))
    missing_vids = gt_keys - pred_keys
    if missing_vids:
        print(f"[WARN] {len(missing_vids)} ground-truth videos have no predictions — "
              "they will count as all-zero windows.")
    extra_vids = pred_keys - gt_keys
    if extra_vids:
        print(f"[WARN] {len(extra_vids)} submitted videos not in ground truth — ignored.")
        pred_df = pred_df[pred_df.apply(
            lambda r: (r.participant_id, r.video_id) in gt_keys, axis=1)]

    # ── WINDOW-LEVEL ─────────────────────────────────────────────────────────
    # Join GT video label onto every window row
    win = pred_df.merge(gt_video[["participant_id", "video_id", "y_true"]],
                        on=["participant_id", "video_id"], how="left")

    y_prob_win = win["y_prob_1"].values if has_proba else None
    m_win = compute_metrics(win["y_true"].values, win["y_pred"].values,
                             y_prob_win, level="window")

    # ── VIDEO-LEVEL — majority vote across windows ────────────────────────────
    vid_pred = (win.groupby(["participant_id", "video_id"])["y_pred"]
                   .agg(lambda x: int(Counter(x).most_common(1)[0][0]))
                   .reset_index().rename(columns={"y_pred": "y_pred_vid"}))
    vid = vid_pred.merge(gt_video[["participant_id", "video_id", "y_true"]],
                         on=["participant_id", "video_id"])

    y_prob_vid = None
    if has_proba:
        vid_prob = (win.groupby(["participant_id", "video_id"])["y_prob_1"]
                       .mean().reset_index()
                       .rename(columns={"y_prob_1": "y_prob_1_vid"}))
        vid = vid.merge(vid_prob, on=["participant_id", "video_id"])
        y_prob_vid = vid["y_prob_1_vid"].values

    m_vid = compute_metrics(vid["y_true"].values, vid["y_pred_vid"].values,
                             y_prob_vid, level="video")

    # ── TEMPORAL METRICS ──────────────────────────────────────────────────────
    det_pct, avg_fnr = compute_temporal_metrics(pred_df, gt_video, ws, sl)

    # ── REPORT ────────────────────────────────────────────────────────────────
    meta = TRACK_META[args.track]
    print_report(args.track, ws, sl, m_win, m_vid, det_pct, avg_fnr,
                 meta["label_0"], meta["label_1"])

    if args.out:
        vid["correct"] = (vid["y_true"] == vid["y_pred_vid"]).astype(int)
        vid.to_csv(args.out, index=False)
        print(f"[INFO] Per-video results saved to {args.out}")


if __name__ == "__main__":
    main()