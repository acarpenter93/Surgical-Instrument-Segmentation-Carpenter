"""
SAR-RARP50 Preprocessing Script
================================
Extracts frames from video_left.avi at 1Hz (matching segmentation mask freq),
pairs them with segmentation masks, and saves train/val/test split manifests.

Handles split video folders like video_11_1, video_11_2 correctly —
both parts are treated as belonging to video 11's split assignment.

Dataset layout:
    train_root/          videos 1-40  (sar-rarp50-train-set Kaggle download)
        video_01/
        video_11_1/
        video_11_2/
        video_15_1/
        video_15_2/
        video_17_1/
        video_17_2/
        video_29_1/
        video_29_2/
        ...
        video_40/

    test_root/           videos 41-50 (sar-rarp50-test-set Kaggle download)
        video_41/
        ...
        video_50/

Split strategy (video-level to prevent temporal data leakage):
    Train : videos  1-36  → from train_root
    Val   : videos 37-40  → from train_root
    Test  : videos 41-50  → from test_root

Usage:
    python preprocess_sar_rarp50.py \\
        --train_root /usr/project/xtmp/acc123/sar-rarp50/train-set \\
        --test_root  /usr/project/xtmp/acc123/sar-rarp50/test-set \\
        --output_dir /home/users/acc123/InstrumentSegmentationSurgery/processed \\
        --verify
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

# Split definitions (video-level, prevents temporal data leakage)

TRAIN_VIDEO_NUMS = set(range(1,  37))   # 1-36
VAL_VIDEO_NUMS   = set(range(37, 41))   # 37-40
TEST_VIDEO_NUMS  = set(range(41, 51))   # 41-50

NUM_CLASSES = 10
CLASS_NAMES = {
    0: "Background",
    1: "Tool clasper",
    2: "Tool wrist",
    3: "Tool shaft",
    4: "Suturing needle",
    5: "Thread",
    6: "Suction tool",
    7: "Needle holder",
    8: "Clamps",
    9: "Catheter",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TARGET_HEIGHT = 512
TARGET_WIDTH  = 512


# Video folder name parsing
# Handles: video_01, video_11_1, video_11_2, video_29_2, etc.

def parse_video_num(folder_name: str) -> int | None:
    """
    Extract base video number from folder name.
        video_01   -> 1
        video_11_1 -> 11
        video_29_2 -> 29
    Returns None if parsing fails.
    """
    name  = folder_name.replace("video_", "")
    parts = name.split("_")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def get_split(folder_name: str) -> str | None:
    """
    Returns 'train', 'val', 'test', or None.
    Split parts (video_11_1, video_11_2) both map to video 11's split.
    """
    num = parse_video_num(folder_name)
    if num is None:
        return None
    if num in TRAIN_VIDEO_NUMS:
        return "train"
    if num in VAL_VIDEO_NUMS:
        return "val"
    if num in TEST_VIDEO_NUMS:
        return "test"
    return None


# Frame extraction

def get_mask_frame_numbers(seg_dir: Path) -> list:
    """Parse frame numbers from segmentation mask filenames."""
    nums = []
    for f in sorted(seg_dir.glob("*.png")):
        try:
            nums.append(int(f.stem))
        except ValueError:
            pass
    return sorted(nums)


def extract_frames_for_video(video_path: Path, seg_dir: Path,
                              output_dir: Path,
                              target_h: int = TARGET_HEIGHT,
                              target_w: int = TARGET_WIDTH) -> list:
    """
    Extract only the frames that have a matching segmentation mask.
    Saves resized RGB frames as .jpg and resized masks as _mask.png.
    Uses INTER_NEAREST for masks to preserve class label values.
    Returns list of (frame_path, mask_path) pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_frame_numbers = get_mask_frame_numbers(seg_dir)

    if not mask_frame_numbers:
        print(f"  [WARN] No masks found in {seg_dir}")
        return []

    if not video_path.exists():
        print(f"  [WARN] Video not found: {video_path} — skipping")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return []

    frame_set = set(mask_frame_numbers)
    pairs     = []
    frame_idx = 0
    pbar      = tqdm(total=len(mask_frame_numbers),
                     desc="  Extracting frames", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_set:
            frame_out = output_dir / f"{frame_idx:09d}.jpg"
            mask_out  = output_dir / f"{frame_idx:09d}_mask.png"
            mask_src  = seg_dir    / f"{frame_idx:09d}.png"

            # Only write if not already extracted (safe to re-run)
            if not frame_out.exists():
                resized = cv2.resize(frame, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(frame_out), resized,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

            if not mask_out.exists() and mask_src.exists():
                mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (target_w, target_h),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mask_out), mask)

            if frame_out.exists() and mask_out.exists():
                pairs.append((str(frame_out), str(mask_out)))

            pbar.update(1)
            frame_set.discard(frame_idx)
            if not frame_set:
                break   
                # all masks accounted for, stop reading

        frame_idx += 1

    cap.release()
    pbar.close()
    return pairs



# Class imbalance analysis


def compute_class_distribution(mask_paths: list, sample_n: int = 300) -> dict:
    """Sample up to sample_n masks and compute pixel-level class distribution."""
    pixel_counts = Counter()
    sampled = mask_paths[:sample_n]

    for mp in tqdm(sampled, desc="Computing class distribution"):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pixel_counts[int(cls)] += int(cnt)

    total = sum(pixel_counts.values())
    distribution = {}
    for cls_id in range(NUM_CLASSES):
        count = pixel_counts.get(cls_id, 0)
        distribution[cls_id] = {
            "class_name":  CLASS_NAMES[cls_id],
            "pixel_count": count,
            "percentage":  round(100.0 * count / total, 4) if total > 0 else 0.0,
        }
    return distribution


def compute_class_weights(distribution: dict) -> dict:
    """
    Inverse-frequency weights for CrossEntropyLoss(weight=...).
    weight_i = total / (num_classes * count_i)
    Normalized so mean weight = 1.
    """
    counts  = np.array([distribution[i]["pixel_count"]
                        for i in range(NUM_CLASSES)], dtype=np.float64)
    counts  = np.clip(counts, 1, None)
    total   = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return {i: round(float(weights[i]), 6) for i in range(NUM_CLASSES)}


# main pipeline

def build_manifest(train_root: Path, test_root: Path,
                   output_dir: Path,
                   target_h: int = TARGET_HEIGHT,
                   target_w: int = TARGET_WIDTH):
    """
    1. Scan train_root for videos 1-40 → assign to train/val splits
    2. Scan test_root  for videos 41-50 → assign to test split
    3. Extract frames + masks for each video
    4. Save split manifests as JSON
    5. Compute class distribution + inverse-frequency weights
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    splits = {"train": [], "val": [], "test": []}

    # Process train_root (videos 1-40)
    print(f"\nScanning train root: {train_root}")
    train_video_dirs = sorted(train_root.glob("video_*"))
    print(f"Found {len(train_video_dirs)} video folders")

    for video_dir in train_video_dirs:
        folder_name = video_dir.name
        split = get_split(folder_name)

        if split is None:
            print(f"  [SKIP] Unrecognized folder: {folder_name}")
            continue

        if split == "test":
            # Videos 41-50 come from test_root, skip any in train_root
            print(f"  [SKIP] {folder_name} belongs to test split, "
                  f"will be loaded from test_root")
            continue

        seg_dir    = video_dir / "segmentation"
        video_path = video_dir / "video_left.avi"
        frame_out  = frames_dir / folder_name

        print(f"\n  [{split.upper()}] {folder_name} "
              f"(video #{parse_video_num(folder_name)})")

        if not seg_dir.exists():
            print(f"    [WARN] No segmentation/ folder, skipping")
            continue

        pairs = extract_frames_for_video(
            video_path, seg_dir, frame_out, target_h, target_w
        )
        splits[split].extend(pairs)
        print(f"    → {len(pairs)} frame/mask pairs")

    #  Process test_root (videos 41-50) 
    print(f"\nScanning test root: {test_root}")
    test_video_dirs = sorted(test_root.glob("video_*"))
    print(f"Found {len(test_video_dirs)} video folders")

    for video_dir in test_video_dirs:
        folder_name = video_dir.name
        split = get_split(folder_name)

        if split != "test":
            print(f"  [SKIP] {folder_name} — not in test range (41-50)")
            continue

        seg_dir    = video_dir / "segmentation"
        video_path = video_dir / "video_left.avi"
        frame_out  = frames_dir / folder_name

        print(f"\n  [TEST] {folder_name} "
              f"(video #{parse_video_num(folder_name)})")

        if not seg_dir.exists():
            print(f"    [WARN] No segmentation/ folder, skipping")
            continue

        pairs = extract_frames_for_video(
            video_path, seg_dir, frame_out, target_h, target_w
        )
        splits["test"].extend(pairs)
        print(f"    → {len(pairs)} frame/mask pairs")

    # save manifests 
    print(f"\n{'='*50}")
    print("Split summary:")
    for split_name, pairs in splits.items():
        manifest = [{"frame": p[0], "mask": p[1]} for p in pairs]
        out_path = output_dir / f"{split_name}_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  {split_name:<6}: {len(manifest):>5} samples → {out_path}")

    total = sum(len(p) for p in splits.values())
    print(f"  {'TOTAL':<6}: {total:>5} samples")

    # Class dist (on training set) 
    print(f"\nComputing class distribution on training set...")
    train_mask_paths = [p[1] for p in splits["train"]]

    if not train_mask_paths:
        print("[WARN] No training masks found — skipping stats")
        return

    distribution = compute_class_distribution(train_mask_paths, sample_n=300)
    weights      = compute_class_weights(distribution)

    print(f"\n{'Class':<5} {'Name':<20} {'%':>8}  {'Weight':>8}")
    print("-" * 50)
    for cls_id, info in distribution.items():
        print(f"  {cls_id:<3} {info['class_name']:<20} "
              f"{info['percentage']:>7.3f}%  {weights[cls_id]:>8.4f}")

    stats = {
        "split_ratios": {
            "train": len(splits["train"]),
            "val":   len(splits["val"]),
            "test":  len(splits["test"]),
        },
        "split_strategy": (
            "Video-level split — prevents temporal data leakage. "
            "Split video folders (e.g. video_11_1, video_11_2) are "
            "assigned to the same split as their base video number."
        ),
        "train_video_nums": sorted(TRAIN_VIDEO_NUMS),
        "val_video_nums":   sorted(VAL_VIDEO_NUMS),
        "test_video_nums":  sorted(TEST_VIDEO_NUMS),
        "image_resolution": f"{target_h}x{target_w}",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std":  IMAGENET_STD,
            "note": "ImageNet stats; applied in Dataset __getitem__"
        },
        "class_distribution":      distribution,
        "class_weights_for_loss":  weights,
        "num_classes":             NUM_CLASSES,
        "class_names":             CLASS_NAMES,
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDataset stats saved → {stats_path}")



# Sanity check


def verify_manifests(output_dir: Path, n_samples: int = 5):
    """Load a few samples from each manifest and confirm shapes are good."""
    for split in ["train", "val", "test"]:
        manifest_path = output_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"[WARN] {manifest_path} not found")
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"\nVerifying {split} ({len(manifest)} samples)...")
        indices = np.random.choice(len(manifest),
                                   min(n_samples, len(manifest)),
                                   replace=False)
        all_ok = True
        for i in indices:
            item  = manifest[i]
            frame = cv2.imread(item["frame"])
            mask  = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                print(f"  [FAIL] Could not read sample {i}: {item}")
                all_ok = False
                continue
            unique_classes = np.unique(mask).tolist()
            print(f"  Sample {i}: frame={frame.shape}, "
                  f"mask={mask.shape}, classes={unique_classes}")

        if all_ok:
            print(f"  [{split}] All samples OK")



# CLI


def parse_args():
    p = argparse.ArgumentParser(
        description="SAR-RARP50 Preprocessing Script"
    )
    p.add_argument(
        "--train_root", type=str,
        default="/usr/project/xtmp/acc123/sar-rarp50/train-set",
        help="Path to sar-rarp50-train-set (videos 1-40)",
    )
    p.add_argument(
        "--test_root", type=str,
        default="/usr/project/xtmp/acc123/sar-rarp50/test-set",
        help="Path to sar-rarp50-test-set (videos 41-50)",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="/home/users/acc123/InstrumentSegmentationSurgery/processed",
        help="Output director",
    )
    p.add_argument("--height",  type=int, default=TARGET_HEIGHT)
    p.add_argument("--width",   type=int, default=TARGET_WIDTH)
    p.add_argument("--verify",  action="store_true",
                   help="Sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_root = Path(args.train_root)
    test_root  = Path(args.test_root)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("SAR-RARP50 Preprocessing")
    print("=" * 60)
    print(f"Train root : {train_root}")
    print(f"Test root  : {test_root}")
    print(f"Output dir : {output_dir}")
    print(f"Resolution : {args.height}x{args.width}")
    print(f"Train videos : {sorted(TRAIN_VIDEO_NUMS)}")
    print(f"Val videos   : {sorted(VAL_VIDEO_NUMS)}")
    print(f"Test videos  : {sorted(TEST_VIDEO_NUMS)}")
    print("=" * 60)

    build_manifest(train_root, test_root, output_dir,
                   args.height, args.width)

    if args.verify:
        verify_manifests(output_dir)

    print("\nDone!--processed_dir "
          f"{output_dir} --visualize")