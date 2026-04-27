"""
evaluate2.py — Additional plots for SAR-RARP50 Segmentation
=============================================================
Generates:
  1. Per-class Dice coefficient bar chart (per run)
  2. Pixel accuracy bar chart (all models compared)
  3. Input frame + GT mask + predicted segmentation for each model

Usage:
    # Per-class Dice + pixel accuracy for all runs
    python evaluate2.py --mode per_class \
        --run_dirs runs/A runs/B runs/C \
        --processed_dir ./processed

    # Prediction examples for each model (input + GT + prediction side by side)
    python evaluate2.py --mode predict \
        --run_dirs runs/A runs/B runs/C \
        --processed_dir ./processed \
        --n_samples 5
"""

import json
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import (get_dataloaders, load_class_weights,
                     NUM_CLASSES, CLASS_NAMES, denormalize,
                     mask_to_color, PALETTE)
from models import build_model, CompoundLoss
from train  import SegmentationMetrics, evaluate



# Helpers


def load_model_from_run(run_dir, device):
    run_dir   = Path(run_dir)
    config    = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / "best_model.pth"
    model = build_model(
        name=config["model"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_p=config.get("dropout", 0.1),
        segformer_backbone=config.get("segformer_backbone", "nvidia/mit-b2"),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, config


def friendly_label(config):
    model  = config.get("model", "?")
    opt    = config.get("optimizer", "?")
    frozen = config.get("freeze_encoder", False)
    ce_w   = config.get("ce_weight", 0.5)
    dice_w = config.get("dice_weight", 0.5)
    label  = f"{model}\n{opt}"
    if frozen:
        label += "\nfrozen"
    if ce_w == 1.0 and dice_w == 0.0:
        label += "\nCE-only"
    return label


def load_test_results(run_dir):
    path = Path(run_dir) / "test_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())



# 1. Per-class Dice bar chart


def plot_per_class_dice(results, title, save_path):
    """Bar chart of per-class Dice score for a single model."""
    dice_vals   = results["dice_per_class"]
    valid       = results["valid_classes"]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = [PALETTE[i] / 255.0 for i in range(NUM_CLASSES)]
    x       = np.arange(NUM_CLASSES)

    bars = ax.bar(x,
                  [v if not np.isnan(v) else 0 for v in dice_vals],
                  color=colors, edgecolor="black", linewidth=0.5)

    for i, (bar, val, v) in enumerate(zip(bars, dice_vals, valid)):
        if v and not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                    "N/A", ha="center", va="bottom", fontsize=7, color="grey")

    mean_dice = results["mdice"]
    ax.axhline(mean_dice, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean Dice = {mean_dice:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontweight="bold")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")



# 2. Pixel accuracy comparison bar chart


def plot_pixel_accuracy_comparison(run_results, save_path):
    """
    Grouped bar chart comparing pixel accuracy across all models.
    Also shows mIoU and Dice for reference.
    """
    labels = [r["label"] for r in run_results]
    accs   = [r["pixel_acc"] for r in run_results]
    mious  = [r["miou"]      for r in run_results]
    dices  = [r["mdice"]     for r in run_results]

    x  = np.arange(len(labels))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.5), 6))

    b1 = ax.bar(x - w, mious, w, label="mIoU",       color="#2196F3", alpha=0.9)
    b2 = ax.bar(x,     dices, w, label="Dice",        color="#4CAF50", alpha=0.9)
    b3 = ax.bar(x + w, accs,  w, label="Pixel Acc",   color="#FF9800", alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison — mIoU / Dice / Pixel Accuracy",
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")



# 3. Per-class Dice comparison across all models


def plot_per_class_dice_comparison(run_results, save_path):
    """Grouped bar chart of per-class Dice for all models side by side."""
    fig, ax = plt.subplots(figsize=(16, 6))
    x       = np.arange(NUM_CLASSES)
    w       = 0.8 / len(run_results)
    colors  = plt.cm.Set1(np.linspace(0, 1, len(run_results)))

    for i, (r, color) in enumerate(zip(run_results, colors)):
        dice_vals = [v if not np.isnan(v) else 0
                     for v in r["dice_per_class"]]
        offset    = (i - len(run_results) / 2 + 0.5) * w
        ax.bar(x + offset, dice_vals, w,
               label=r["label"].replace("\n", "/"),
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Dice Score — All Models", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")



# 4. Prediction visualization — input + GT + prediction for each model


@torch.no_grad()
def get_predictions(model, loader, device, n_samples=5):
    """Grab n_samples batches and return images, GT masks, predictions."""
    model.eval()
    samples = []
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        with autocast(enabled=True):
            logits = model(images)
        preds = logits.argmax(dim=1)
        for i in range(images.shape[0]):
            samples.append({
                "image": images[i].cpu(),
                "mask":  masks[i].cpu(),
                "pred":  preds[i].cpu(),
            })
            if len(samples) >= n_samples:
                return samples
    return samples


def plot_predictions_single_model(samples, model_label, save_path):
    """
    For one model: n rows × 3 cols (Input | Ground Truth | Prediction).
    Red overlay on misclassified pixels in prediction column.
    """
    n   = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))
    fig.suptitle(f"Predictions — {model_label}",
                 fontsize=13, fontweight="bold")

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Frame", "Ground Truth", "Prediction"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight="bold")

    for row, rec in enumerate(samples):
        img_np  = denormalize(rec["image"])
        gt_np   = mask_to_color(rec["mask"].numpy())
        pred_np = mask_to_color(rec["pred"].numpy())
        errors  = (rec["pred"] != rec["mask"]).numpy()

        axes[row, 0].imshow(img_np);   axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_np);    axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_np)

        # Red overlay on wrong pixels
        err_rgba = np.zeros((*errors.shape, 4), dtype=np.float32)
        err_rgba[errors] = [1, 0, 0, 0.35]
        axes[row, 2].imshow(err_rgba)
        axes[row, 2].axis("off")

        # Row label with IoU
        pred_flat = rec["pred"].numpy().flatten()
        mask_flat = rec["mask"].numpy().flatten()
        ious = []
        for c in range(NUM_CLASSES):
            inter = ((pred_flat == c) & (mask_flat == c)).sum()
            union = ((pred_flat == c) | (mask_flat == c)).sum()
            if union > 0:
                ious.append(inter / union)
        sample_miou = float(np.mean(ious)) if ious else 0.0
        axes[row, 0].set_ylabel(f"mIoU={sample_miou:.3f}",
                                fontsize=9, rotation=0,
                                labelpad=60, va="center")

    # Class legend
    patches = [mpatches.Patch(color=PALETTE[c] / 255.0,
                              label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_predictions_all_models(all_model_samples, save_path, sample_idx=0):
    """
    For a single input frame: one row per model showing
    Input | Ground Truth | Prediction.
    Lets you directly compare how each model segments the same image.
    """
    n_models = len(all_model_samples)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, n_models * 4))
    fig.suptitle(f"Same Frame — All Models Compared",
                 fontsize=13, fontweight="bold")

    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Frame", "Ground Truth", "Prediction"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight="bold")

    for row, (model_label, samples) in enumerate(all_model_samples):
        rec     = samples[sample_idx]
        img_np  = denormalize(rec["image"])
        gt_np   = mask_to_color(rec["mask"].numpy())
        pred_np = mask_to_color(rec["pred"].numpy())
        errors  = (rec["pred"] != rec["mask"]).numpy()

        axes[row, 0].imshow(img_np);  axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_np);   axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_np)

        err_rgba = np.zeros((*errors.shape, 4), dtype=np.float32)
        err_rgba[errors] = [1, 0, 0, 0.35]
        axes[row, 2].imshow(err_rgba)
        axes[row, 2].axis("off")

        axes[row, 0].set_ylabel(model_label.replace("\n", "/"),
                                fontsize=9, rotation=0,
                                labelpad=80, va="center")

    patches = [mpatches.Patch(color=PALETTE[c] / 255.0,
                              label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")



# Mode: per_class — per-class Dice + pixel accuracy for all runs


def mode_per_class(args):
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        config  = json.loads((run_dir / "config.json").read_text())
        results = load_test_results(run_dir)
        if results is None:
            print(f"[SKIP] No test_results.json: {run_dir.name}")
            continue

        label = friendly_label(config)
        results["label"] = label

        # Per-class Dice bar for this model
        plot_per_class_dice(
            results,
            title=f"{label} — Per-class Dice Score",
            save_path=save_dir / f"dice_{run_dir.name}.png",
        )
        run_results.append(results)

    if not run_results:
        print("No completed runs found.")
        return

    # Pixel accuracy + mIoU + Dice comparison across all models
    plot_pixel_accuracy_comparison(
        run_results,
        save_path=save_dir / "pixel_accuracy_comparison.png",
    )

    # Per-class Dice comparison across all models
    plot_per_class_dice_comparison(
        run_results,
        save_path=save_dir / "per_class_dice_comparison.png",
    )

    # Print summary table
    print(f"\n{'Model':<35} {'mIoU':>8} {'Dice':>8} {'PixAcc':>8}")
    print("─" * 65)
    for r in run_results:
        print(f"  {r['label'].replace(chr(10), '/') :<33} "
              f"{r['miou']:>8.4f} {r['mdice']:>8.4f} "
              f"{r['pixel_acc']:>8.4f}")



# Mode: predict — input + GT + prediction examples for each model


def mode_predict(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.output_dir) / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_model_samples = []

    for run_dir in args.run_dirs:
        run_dir       = Path(run_dir)
        model, config = load_model_from_run(run_dir, device)
        label         = friendly_label(config)

        loaders = get_dataloaders(
            processed_dir=args.processed_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=config.get("img_size", 384),
            splits=["val"],
        )

        print(f"Getting predictions: {label}")
        samples = get_predictions(model, loaders["val"], device,
                                  n_samples=args.n_samples)

        plot_predictions_single_model(
            samples,
            model_label=label.replace("\n", "/"),
            save_path=save_dir / f"predictions_{run_dir.name}.png",
        )

        all_model_samples.append((label, samples))
        del model
        torch.cuda.empty_cache()

    if len(all_model_samples) > 1:
        for sample_idx in range(min(3, args.n_samples)):
            plot_predictions_all_models(
                all_model_samples,
                save_path=save_dir / f"all_models_frame_{sample_index}.png",
                sample_idx=sample_index,
            )

    print(f"\nAll prediction plots saved to: {save_dir}")



# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["per_class", "predict"])
    p.add_argument("--run_dirs",      nargs="+", default=[])
    p.add_argument("--processed_dir", default="./processed")
    p.add_argument("--output_dir",    default="./comparison2")
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--n_samples",     type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "per_class":
        mode_per_class(args)
    elif args.mode == "predict":
        mode_predict(args)