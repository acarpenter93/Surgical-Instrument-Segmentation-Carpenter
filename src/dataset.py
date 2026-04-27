"""
SAR-RARP50 PyTorch Dataset & DataLoader
========================================
Loads frame/mask pairs from manifests produced by preprocess_sar_rarp50.py.
Applies augmentation pipelines (train) and normalization (all splits).

Usage:
    from dataset import get_dataloaders, SARRARP50Dataset

    loaders = get_dataloaders(
        processed_dir="./processed",
        batch_size=8,
        num_workers=4,
        img_size=512,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]
"""

import json
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

 
# Constants
 

NUM_CLASSES = 10

CLASS_NAMES = [
    "Background",       # 0
    "Tool clasper",     # 1
    "Tool wrist",       # 2
    "Tool shaft",       # 3
    "Suturing needle",  # 4
    "Thread",           # 5
    "Suction tool",     # 6
    "Needle holder",    # 7
    "Clamps",           # 8
    "Catheter",         # 9
]

# ImageNet stats — matches pretrained SegFormer & ResNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


 
# Augmentation pipelines
 
# All transforms use albumentations so spatial transforms apply
# identically to both the image and the mask.
#
# TRAIN: 6 data augmentation techniques
#   1. Random horizontal flip
#   2. Random vertical flip
#   3. Random rotation 
#   4. Random resized crop (scale jitter)
#   5. Color jitter (brightness, contrast, saturation, hue)
#   6. Gaussian blur (simulates endoscope lens blur / motion)
#   + Normalization + ToTensor
#
# VAL/TEST: only normalization + ToTensor 
 

def build_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7,
        ),
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5,
        ),

        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
            p=0.7,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # make sure output is always correct size
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR),

        # norma & to tensor 
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


 
# Dataset
 
class SARRARP50Dataset(Dataset):
    """
    PyTorch Dataset for SAR-RARP50 semantic segmentation.

    Args:
        manifest_path : Path to JSON manifest (output of preprocess_sar_rarp50.py)
        transforms    : Albumentations Compose pipeline
        img_size      : Target H/W (used as fallback resize if transform is None)

    Returns (per __getitem__):
        image : FloatTensor [3, H, W]  — normalized RGB frame
        mask  : LongTensor  [H, W]     — class indices 0..NUM_CLASSES-1
    """

    def __init__(self, manifest_path: str | Path, transforms: A.Compose = None,
                 img_size: int = 512):
        self.img_size   = img_size
        self.transforms = transforms

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self.samples = json.load(f)  

        if len(self.samples) == 0:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        print(f"Loaded {len(self.samples)} samples from {manifest_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        #load image (BGR -> RGB) 
        image = cv2.imread(item["frame"])
        if image is None:
            raise IOError(f"Could not read frame: {item['frame']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #load mask (grayscale, values 0-9) 
        mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask: {item['mask']}")

        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)

        # apply transforms   
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]   
            # [3, H, W]
            mask  = augmented["mask"]    
            # [H, W]
        else:
            # Fallback -- min. processing w/o augmentations
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = cv2.resize(mask, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)
            mask  = torch.from_numpy(mask)

        return image.float(), mask.long()

    def get_sample_path(self, idx: int) -> dict:
        """Return raw paths for a given index (useful for error analysis)."""
        return self.samples[idx]


 
# DataLoader factory
 

def get_dataloaders(
    processed_dir: str | Path,
    batch_size: int     = 8,
    num_workers: int    = 4,
    img_size: int       = 512,
    pin_memory: bool    = True,
    splits: list[str]   = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """
    Build DataLoaders for all requested splits.

    Args:
        processed_dir : Directory containing *_manifest.json files
        batch_size    : Samples per batch
        num_workers   : Parallel workers for data loading
        img_size      : Image resolution fed to model
        pin_memory    : Faster GPU transfer (set False if no CUDA)
        splits        : Which splits to load

    Returns:
        dict with keys "train", "val", "test" mapping to DataLoaders
    """
    processed_dir = Path(processed_dir)
    loaders = {}

    transform_map = {
        "train": build_train_transforms(img_size),
        "val":   build_val_transforms(img_size),
        "test":  build_val_transforms(img_size),
    }

    for split in splits:
        manifest_path = processed_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            print(f"[WARN] Manifest not found for split '{split}': {manifest_path}")
            continue

        dataset = SARRARP50Dataset(
            manifest_path=manifest_path,
            transforms=transform_map[split],
            img_size=img_size,
        )

        shuffle    = (split == "train")
        drop_last  = (split == "train")  
        # avoids small batch edge cases

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

        loaders[split] = loader
        print(f"  [{split}] {len(dataset)} samples, "
              f"{len(loader)} batches (batch_size={batch_size})")

    return loaders


 
# Class weights loader (for weighted CrossEntropyLoss)
 

def load_class_weights(processed_dir: str | Path,
                       device: str = "cpu") -> torch.Tensor:
    """
    Load inverse-frequency class weights computed during preprocessing.
    Pass to: nn.CrossEntropyLoss(weight=load_class_weights(..., device=device))

    Returns: FloatTensor of shape [NUM_CLASSES]
    """
    stats_path = Path(processed_dir) / "dataset_stats.json"
    if not stats_path.exists():
        print("[WARN] dataset_stats.json not found, using uniform weights")
        return torch.ones(NUM_CLASSES, dtype=torch.float32).to(device)

    with open(stats_path) as f:
        stats = json.load(f)

    weights_dict = stats.get("class_weights_for_loss", {})
    weights = [weights_dict.get(str(i), 1.0) for i in range(NUM_CLASSES)]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"Class weights loaded from {stats_path.name}:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"  [{i}] {name:<20} weight={w:.4f}")

    return weight_tensor


 
# Visualization utility stuff
 

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization for visualization.
    Input:  FloatTensor [3, H, W]
    Output: uint8 numpy array [H, W, 3] in RGB
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


# Color palette for mask visualization (one color per class)
PALETTE = np.array([
    [0,   0,   0  ],  # 0 Background      — black
    [255, 0,   0  ],  # 1 Tool clasper    — red
    [0,   255, 0  ],  # 2 Tool wrist      — green
    [0,   0,   255],  # 3 Tool shaft      — blue
    [255, 255, 0  ],  # 4 Suturing needle — yellow
    [255, 0,   255],  # 5 Thread          — magenta
    [0,   255, 255],  # 6 Suction tool    — cyan
    [128, 0,   128],  # 7 Needle holder   — purple
    [255, 128, 0  ],  # 8 Clamps          — orange
    [128, 255, 128],  # 9 Catheter        — light green
], dtype=np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale label mask [H, W] to a color image [H, W, 3].
    """
    color = PALETTE[np.clip(mask, 0, NUM_CLASSES - 1)]
    return color


def visualize_batch(images: torch.Tensor, masks: torch.Tensor,
                    preds: torch.Tensor = None, n: int = 4,
                    save_path: str = None):
    """
    Visualize a batch of images + ground truth masks (+ optional predictions).

    Args:
        images    : [B, 3, H, W] normalized tensor
        masks     : [B, H, W] long tensor (ground truth)
        preds     : [B, H, W] long tensor (model predictions, optional)
        n         : number of samples to show
        save_path : if provided, save figure instead of showing it
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = min(n, images.shape[0])
    cols = 3 if preds is not None else 2
    fig, axes = plt.subplots(n, cols, figsize=(cols * 5, n * 4))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_np   = denormalize(images[i])
        mask_np  = masks[i].cpu().numpy()
        mask_rgb = mask_to_color(mask_np)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Frame")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        if preds is not None:
            pred_np  = preds[i].cpu().numpy()
            pred_rgb = mask_to_color(pred_np)
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

    patches = [mpatches.Patch(color=PALETTE[c] / 255.0, label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()


 
# smoke test
 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="./processed")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--img_size",      type=int, default=512)
    parser.add_argument("--visualize",     action="store_true",
                        help="Save a visualization of one batch")
    args = parser.parse_args()

    print("Building DataLoaders...")
    loaders = get_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    for split, loader in loaders.items():
        images, masks = next(iter(loader))
        print(f"\n[{split}] batch shapes: images={tuple(images.shape)}, "
              f"masks={tuple(masks.shape)}")
        print(f"  image dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
        print(f"  mask  dtype={masks.dtype},  unique classes={masks.unique().tolist()}")

        if args.visualize and split == "train":
            visualize_batch(images, masks, n=2,
                            save_path=f"./sample_{split}_batch.png")

    print("\nLoading class weights...")
    weights = load_class_weights(args.processed_dir,
                                 device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Weight tensor shape: {weights.shape}")
    print("\nDataset smoke test passed!")
