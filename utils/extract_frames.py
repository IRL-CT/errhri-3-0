#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAD Dataset — Frame extraction + NPY conversion at 5 fps.

For each video in trainval/ and test/:
  - Extracts frames at 5 fps with ffmpeg (piped directly to numpy, no temp files)
  - Saves PNG frames to  {split}_frames/{participant_id}/
  - Saves NPY arrays to  {split}_npy/{participant_id}/
  - Writes label_data.csv to each {split}_frames/ and {split}_npy/ directory

Frame naming convention (matches badidea / badnet infrastructure):
  q_{qid}_main_{label}_5fps_frame{n:04d}.png / .npy

Label CSV columns:  participant_id, q_id, label  (one row per frame)

Usage:
  python extract_frames.py                         # both splits
  python extract_frames.py --split trainval
  python extract_frames.py --split test
  python extract_frames.py --no_npy               # frames only
  python extract_frames.py --no_frames            # NPY only (frames must exist)
"""

import os
import io
import re
import csv
import argparse
import subprocess
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATS_CSV  = os.path.join(os.path.dirname(BASE_DIR),
                           "errhri-3-0", "baddataset_stats.csv")
TARGET_FPS = 5
IMG_SIZE   = 224   # NPY resize target (square)

NORMALIZE = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stats(stats_csv):
    """Return dict: (participant_id, video_name) -> label, grouped by split."""
    splits = {"trainval": {}, "test": {}}
    with open(stats_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid  = str(row["participant_id"])
            name = row["video_name"]          # e.g. QID106
            lbl  = int(row["label"])
            spl  = row["split"]
            if spl in splits:
                splits[spl][(pid, name)] = lbl
    return splits


def qid_to_int(video_name):
    """QID106 -> 106"""
    m = re.search(r"\d+", video_name)
    return int(m.group()) if m else 0


def frame_stem(qid_int, label, frame_idx):
    """q_106_main_1_5fps_frame0001"""
    return f"q_{qid_int}_main_{label}_5fps_frame{frame_idx:04d}"


def extract_frames_pipe(video_path):
    """
    Stream raw RGB24 frames from ffmpeg at TARGET_FPS.
    Yields PIL Images in order.
    """
    # Probe width/height
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    w, h = map(int, probe.stdout.strip().split(","))

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={TARGET_FPS}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    idx = 1
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        yield idx, Image.fromarray(arr)
        idx += 1
    proc.stdout.close()
    proc.wait()


def process_split(split, label_map, base_dir, do_frames, do_npy):
    video_dir   = os.path.join(base_dir, split)
    frames_dir  = os.path.join(base_dir, f"{split}_frames")
    npy_dir     = os.path.join(base_dir, f"{split}_npy")

    frames_csv_rows = []   # (participant_id, q_id, label) per frame
    npy_csv_rows    = []

    participants = sorted(os.listdir(video_dir))
    total_videos = sum(1 for p in participants
                       for _ in os.listdir(os.path.join(video_dir, p))
                       if _.endswith(".mp4"))
    done = 0

    for pid in participants:
        pid_video_dir = os.path.join(video_dir, pid)
        if not os.path.isdir(pid_video_dir):
            continue

        if do_frames:
            os.makedirs(os.path.join(frames_dir, pid), exist_ok=True)
        if do_npy:
            os.makedirs(os.path.join(npy_dir, pid), exist_ok=True)

        videos = sorted(f for f in os.listdir(pid_video_dir) if f.endswith(".mp4"))

        for vid_file in videos:
            # video_name = QID106  (strip _1.mp4)
            video_name = re.sub(r"_\d+\.mp4$", "", vid_file)
            key = (pid, video_name)

            if key not in label_map:
                print(f"  [WARN] no label for {pid}/{vid_file}, skipping")
                continue

            label   = label_map[key]
            qid_int = qid_to_int(video_name)
            q_id    = f"q_{qid_int}"

            video_path   = os.path.join(pid_video_dir, vid_file)
            out_frame_p  = os.path.join(frames_dir, pid)
            out_npy_p    = os.path.join(npy_dir, pid)

            # Check if already done (resume support)
            if do_frames:
                existing_frames = [
                    f for f in os.listdir(out_frame_p)
                    if f.startswith(f"q_{qid_int}_main_{label}_5fps_") and f.endswith(".png")
                ]
            else:
                existing_frames = []

            if do_npy:
                existing_npy = [
                    f for f in os.listdir(out_npy_p)
                    if f.startswith(f"q_{qid_int}_main_{label}_5fps_") and f.endswith(".npy")
                ]
            else:
                existing_npy = []

            # If frames dir already has files for this video, reload them for CSV
            # rather than re-extracting
            if existing_frames and not do_npy:
                for fname in sorted(existing_frames):
                    frames_csv_rows.append((pid, q_id, label))
                done += 1
                continue

            if existing_npy and not do_frames:
                for fname in sorted(existing_npy):
                    npy_csv_rows.append((pid, q_id, label))
                done += 1
                continue

            done += 1
            print(f"  [{done}/{total_videos}] {pid}/{vid_file}  label={label}", flush=True)

            for frame_idx, pil_img in extract_frames_pipe(video_path):
                stem = frame_stem(qid_int, label, frame_idx)

                if do_frames:
                    png_path = os.path.join(out_frame_p, stem + ".png")
                    if not os.path.exists(png_path):
                        pil_img.save(png_path)
                    frames_csv_rows.append((pid, q_id, label))

                if do_npy:
                    npy_path = os.path.join(out_npy_p, stem + ".npy")
                    if not os.path.exists(npy_path):
                        arr = NORMALIZE(pil_img).numpy()   # (3, 224, 224) float32
                        np.save(npy_path, arr)
                    npy_csv_rows.append((pid, q_id, label))

    # Write label CSVs
    if do_frames and frames_csv_rows:
        csv_path = os.path.join(frames_dir, "label_data.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["participant_id", "q_id", "label"])
            w.writerows(frames_csv_rows)
        print(f"\n  Wrote {len(frames_csv_rows)} rows -> {csv_path}")

    if do_npy and npy_csv_rows:
        csv_path = os.path.join(npy_dir, "label_data.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["participant_id", "q_id", "label"])
            w.writerows(npy_csv_rows)
        print(f"  Wrote {len(npy_csv_rows)} rows -> {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract BAD dataset frames at 5 fps")
    parser.add_argument("--split", choices=["trainval", "test", "both"], default="both")
    parser.add_argument("--base_dir", default=BASE_DIR,
                        help="Root data_baddataset directory")
    parser.add_argument("--stats_csv", default=STATS_CSV,
                        help="Path to baddataset_stats.csv")
    parser.add_argument("--no_frames", action="store_true",
                        help="Skip saving PNG frames (NPY only)")
    parser.add_argument("--no_npy", action="store_true",
                        help="Skip saving NPY arrays (frames only)")
    args = parser.parse_args()

    do_frames = not args.no_frames
    do_npy    = not args.no_npy

    if not do_frames and not do_npy:
        parser.error("--no_frames and --no_npy both set — nothing to do.")

    print(f"Loading labels from {args.stats_csv}")
    splits_map = load_stats(args.stats_csv)

    splits_to_run = ["trainval", "test"] if args.split == "both" else [args.split]

    for split in splits_to_run:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")
        process_split(split, splits_map[split], args.base_dir, do_frames, do_npy)

    print("\nDone.")


if __name__ == "__main__":
    main()
