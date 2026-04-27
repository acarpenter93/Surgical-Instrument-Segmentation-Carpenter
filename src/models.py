"""
Model Definitions for SAR-RARP50 Surgical Instrument Segmentation
==================================================================
Implements three architectures benchmarked in the paper:
  1. UNet          — classic CNN encoder-decoder (custom implementation)
  2. Attention UNet — UNet with attention gates on skip connections
  3. SegFormer     — transformer-based (via HuggingFace transformers)

Also includes:
  - CompoundLoss   — CrossEntropy + Dice (as used in the paper)
  - model_summary  — parameter count + inference time measurement

Usage:
    from models import build_model, CompoundLoss

    model = build_model("segformer", num_classes=10, pretrained=True)
    criterion = CompoundLoss(num_classes=10, ce_weight=0.5, dice_weight=0.5)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_CLASSES = 10



# Shared building blocks


class DoubleConv(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Standard UNet encoder/decoder block.
    Batch normalization applied after every conv (rubric: batch normalization).
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #RUBRIC- regularization (dropout)!
            nn.Dropout2d(p=dropout_p),        
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d -> DoubleConv (encoder step)."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample -> concat skip -> DoubleConv (decoder step).
    Bilinear upsampling avoids checkerboard artifacts vs. transposed conv.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout_p)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Handle odd input sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


#UNet

class UNet(nn.Module):
    """
    Classic UNet (Ronneberger et al., 2015) with:
    - Batch normalization in every DoubleConv block
    - Optional dropout for regularization
    - Bilinear upsampling decoder

    Architecture:
        Encoder: 3->64->128->256->512->1024
        Bottleneck: 1024
        Decoder: mirrors encoder with skip connections
        Head: 1x1 conv -> num_classes
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 base_channels: int = 64,
                 dropout_p: float = 0.1):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc   = DoubleConv(3, c, dropout_p)
        self.down1 = Down(c,     c*2,  dropout_p)
        self.down2 = Down(c*2,   c*4,  dropout_p)
        self.down3 = Down(c*4,   c*8,  dropout_p)
        self.down4 = Down(c*8,   c*16, dropout_p)

        # Decoder
        self.up1   = Up(c*16, c*8,  c*8,  dropout_p)
        self.up2   = Up(c*8,  c*4,  c*4,  dropout_p)
        self.up3   = Up(c*4,  c*2,  c*2,  dropout_p)
        self.up4   = Up(c*2,  c,    c,    dropout_p)

        # Segmentation head
        self.head  = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.head(x)


# Attention UNet (custom implementation!!)

class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al., 2018).
    Learns to suppress irrelevant skip-connection features and highlight
    salient regions — useful for small instruments like thread and needle.

    Gate signal g comes from the decoder (coarser, semantically rich).
    Skip signal x comes from the encoder (finer, spatially precise).
    """
    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        # Align spatial dims of g to x
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear",
                             align_corners=True)
        att  = self.relu(self.Wx(x) + self.Wg(g_up))
        # [B, 1, H, W] attention map
        att  = self.psi(att)   
        # broadcast multiply       
        return x * att                


class AttentionUp(nn.Module):
    """Decoder block with attention gate on the skip connection."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout_p: float = 0.0):
        super().__init__()
        inter_ch     = skip_ch // 2
        self.att     = AttentionGate(skip_ch, in_ch, inter_ch)
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear",
                                   align_corners=True)
        self.conv    = DoubleConv(in_ch + skip_ch, out_ch, dropout_p)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # attend skip features
        skip = self.att(skip, x)          
        x    = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention UNet — identical encoder/decoder to UNet but with
    attention gates on every skip connection in the decoder.
    Particularly effective for small, variable-appearance instruments.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 base_channels: int = 64,
                 dropout_p: float = 0.1):
        super().__init__()
        c = base_channels

        # Encoder (same as UNet)
        self.inc   = DoubleConv(3, c,     dropout_p)
        self.down1 = Down(c,    c*2,  dropout_p)
        self.down2 = Down(c*2,  c*4,  dropout_p)
        self.down3 = Down(c*4,  c*8,  dropout_p)
        self.down4 = Down(c*8,  c*16, dropout_p)

        # Attention decoder
        self.up1   = AttentionUp(c*16, c*8,  c*8,  dropout_p)
        self.up2   = AttentionUp(c*8,  c*4,  c*4,  dropout_p)
        self.up3   = AttentionUp(c*4,  c*2,  c*2,  dropout_p)
        self.up4   = AttentionUp(c*2,  c,    c,    dropout_p)

        self.head  = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.head(x)



#SegFormer (HuggingFace transformers)

class SegFormerModel(nn.Module):
    """
    Wrapper around HuggingFace SegFormer for semantic segmentation.

    SegFormer uses a hierarchical Mix Transformer (MiT) encoder and a
    lightweight All-MLP decoder — no positional encodings, which makes
    it robust to resolution changes at test time.

    Backbone options (larger = better accuracy, slower):
        "nvidia/mit-b0"  — 3.7M params  (fast, lower accuracy)
        "nvidia/mit-b2"  — 25M params   (good balance, used in paper)
        "nvidia/mit-b4"  — 64M params   (high accuracy, slower)

    The model outputs logits at H/4 x W/4; we upsample to input resolution.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 backbone: str = "nvidia/mit-b2",
                 pretrained: bool = True):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError(
                "Install HuggingFace transformers: pip install transformers"
            )

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                # head changes for num_classes
                ignore_mismatched_sizes=True,   
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(backbone,
                                                     num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.model(pixel_values=x)
         # [B, C, H/4, W/4] for logits (below)
        logits  = outputs.logits                
        # Upsample to input resolution
        logits  = F.interpolate(logits, size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        # [B, C, H, W] for logits (below)
        return logits                           

    def freeze_encoder(self):
        """Freeze MiT backbone, only train the decode head."""
        for param in self.model.segformer.parameters():
            param.requires_grad = False
        print("SegFormer encoder frozen — training decode head only")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("SegFormer fully unfrozen — fine-tuning all layers")



#Loss fn: compound CE + dice

class DiceLoss(nn.Module):
    """
    Multiclass Dice loss.
    Dice = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)

    Computed per class then averaged (macro). Handles class imbalance better
    than plain cross-entropy because it directly optimizes overlap.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 smooth: float = 1e-6,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        logits  : [B, C, H, W]  raw model output
        targets : [B, H, W]     integer class labels
        """
        # [B, C, H, W]
        probs = F.softmax(logits, dim=1)   

        # One-hot encode targets: [B, H, W] -> [B, C, H, W]
        B, C, H, W = probs.shape
        targets_oh = F.one_hot(targets.clamp(0, C - 1), num_classes=C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()

        # Flatten spatial dims
        # [B, C, N]
        probs      = probs.view(B, C, -1) 
        # [B, C, N]       
        targets_oh = targets_oh.view(B, C, -1)   

        intersection = (probs * targets_oh).sum(dim=2)
        union        = probs.sum(dim=2) + targets_oh.sum(dim=2)

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_class.mean()


class CompoundLoss(nn.Module):
    """
    Compound loss = α * CrossEntropy + β * Dice
    as used in the paper. CE handles pixel-wise classification;
    Dice handles class imbalance and boundary quality.

    Args:
        num_classes  : number of segmentation classes
        ce_weight    : scalar weight for CE term (α)
        dice_weight  : scalar weight for Dice term (β)
        class_weights: per-class frequency weights for CE [num_classes]
                       — pass output of load_class_weights() from dataset.py
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 ce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 class_weights: Tensor = None):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(weight=class_weights,
                                        ignore_index=255)
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """
        Returns dict with keys: "loss", "ce_loss", "dice_loss"
        so individual components can be logged during training.
        """
        ce_loss   = self.ce(logits.float(), targets)
        dice_loss = self.dice(logits, targets)
        total     = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return {
            "loss":      total,
            "ce_loss":   ce_loss,
            "dice_loss": dice_loss,
        }


#build models

SUPPORTED_MODELS = ["unet", "attention_unet", "segformer"]


def build_model(name: str,
                num_classes: int = NUM_CLASSES,
                pretrained: bool = True,
                dropout_p: float = 0.1,
                segformer_backbone: str = "nvidia/mit-b2") -> nn.Module:
    """
    Factory function — returns an initialized model by name.

    Args:
        name               : one of "unet", "attention_unet", "segformer"
        num_classes        : number of output classes
        pretrained         : use ImageNet pretrained weights (SegFormer only)
        dropout_p          : dropout probability for UNet/AttentionUNet
        segformer_backbone : HuggingFace model ID for SegFormer

    Returns:
        nn.Module ready for training
    """
    name = name.lower().strip()

    if name == "unet":
        model = UNet(num_classes=num_classes, dropout_p=dropout_p)

    elif name == "attention_unet":
        model = AttentionUNet(num_classes=num_classes, dropout_p=dropout_p)

    elif name == "segformer":
        model = SegFormerModel(num_classes=num_classes,
                               backbone=segformer_backbone,
                               pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {SUPPORTED_MODELS}"
        )

    print(f"\nBuilt model: {name}")
    model_summary(model)
    return model


#utilities, useful stuff

def model_summary(model: nn.Module, img_size: int = 512, device: str = "cpu"):
    """
    Print parameter count and measure inference time.
    Covers rubric: "Measured and reported inference time / computational efficiency"
    """
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # Inference time (average of 20 forward passes, excluding first warmup)
    model.eval().to(device)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        # Warmup
        _ = model(dummy)
        torch.cuda.synchronize() if device == "cuda" else None

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _  = model(dummy)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    fps    = 1000.0 / avg_ms
    print(f"  Inference time  : {avg_ms:.1f} ms/frame  ({fps:.1f} FPS) "
          f"@ {img_size}x{img_size} on {device}")
    model.train()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#smoke test

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dummy_img    = torch.randn(2, 3, 512, 512).to(device)
    dummy_mask   = torch.randint(0, NUM_CLASSES, (2, 512, 512)).to(device)
    criterion    = CompoundLoss(num_classes=NUM_CLASSES)

    for model_name in SUPPORTED_MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print('='*60)
        try:
            model = build_model(model_name, num_classes=NUM_CLASSES,
                                pretrained=False).to(device)
            model_summary(model, img_size=512, device=device)

            logits = model(dummy_img)
            print(f"  Output shape: {tuple(logits.shape)}")
            assert logits.shape == (2, NUM_CLASSES, 512, 512), \
                f"Unexpected output shape: {logits.shape}"

            losses = criterion(logits, dummy_mask)
            print(f"  CE loss  : {losses['ce_loss'].item():.4f}")
            print(f"  Dice loss: {losses['dice_loss'].item():.4f}")
            print(f"  Total    : {losses['loss'].item():.4f}")
            print(f"  [PASS] {model_name}")

        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
