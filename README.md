# Surgical Instrument Segmentation in Robotic Surgery

Surgical instrument segmentation in robotic-assisted surgery is a critical computer vision challenge, yet it remains difficult due to extreme class imbalance, small instrument size, and heavy occlusion. This project asks: **can transformer-based architectures outperform traditional CNNs for this task, and can optimizer selection further improve upon published results?** To answer this, we reproduce and extend the benchmark from Ameli (2026), training and evaluating UNet, Attention UNet, and SegFormer on the SAR-RARP50 dataset of 50 real robotic prostatectomy surgeries with pixel-level segmentation masks across 10 instrument classes.

---

## What it Does

This project benchmarks three deep learning architectures — UNet, Attention UNet, and SegFormer — for multi-class semantic segmentation of surgical instruments across 10 classes (background + 9 instrument types) in robotic surgery video. The models are trained and evaluated on the SAR-RARP50 dataset, which contains segmentation masks from 50 real robotic prostatectomy operations. The project reproduces the paper's experimental setup (Adam optimizer, lr=1e-4, 384×384 resolution, 10 epochs, batch size 4) and demonstrates an improvement over the paper's baseline by substituting AdamW, which provides better regularization for transformer fine-tuning via decoupled weight decay. Training uses a compound CrossEntropy + Dice loss with inverse-frequency class weights to handle severe class imbalance (background comprises ~85–90% of pixels), mixed precision for efficiency, and DistributedDataParallel for multi-GPU acceleration. An ablation study evaluates the impact of loss function composition (CE-only vs CE+Dice) and backbone fine-tuning strategy (frozen encoder vs full fine-tuning).

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download dataset**
```bash
bash data/download_data.sh
```

**3. Preprocess**
```bash
python src/preprocess_sar_rarp50.py \
    --train_root /path/to/train-set \
    --test_root  /path/to/test-set \
    --output_dir ./processed \
    --verify
```

**4. Train**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/train.py \
    --model segformer --optimizer adamw --lr 1e-4 \
    --processed_dir ./processed --output_dir ./runs \
    --batch_size 4 --img_size 384 --epochs 10
```

**5. Evaluate**
```bash
python src/evaluate.py --mode compare \
    --run_dirs runs/run1 runs/run2 runs/run3 \
    --processed_dir ./processed --output_dir ./comparison
```

See [SETUP.md](SETUP.md) for full instructions.

---

## Video Links

- **Demo Video:** https://drive.google.com/file/d/1EBB6CWNpgzeX2slxdqQxXCbf5sGG_S-A/view?usp=drive_link
- **Technical Walkthrough:** https://drive.google.com/file/d/1HmuwFe3bU86C-RM5Pof2U5B2Qjx6X4iS/view?usp=sharing 

---

## Evaluation

All models trained with: batch size 4, 384×384 resolution, 10 epochs, lr=1e-4.
Paper settings use Adam optimizer. Improvement run uses AdamW.

### Model Comparison (Test Set)

| Model | Optimizer | mIoU | Dice | Pixel Acc |
|---|---|---|---|---|
| UNet | Adam | 0.2871 | 0.3917 | 0.8300 |
| Attention UNet | Adam | 0.2830 | 0.3870 | 0.8299 |
| SegFormer mit-b2 | Adam | 0.2378 | 0.3468 | 0.6512 |
| **SegFormer mit-b2** | **AdamW** | **0.7271** | **0.8314** | **0.9552** |
| SegFormer mit-b2 | AdamW (CE only) | 0.6455 | 0.7623 | 0.9293 |
| SegFormer mit-b2 | AdamW (frozen) | 0.4855 | 0.6179 | 0.8678 |

### Per-class IoU (Test Set)

| Class | UNet | Attn UNet | SegFormer Adam | SegFormer AdamW |
|---|---|---|---|---|
| Background | 0.8582 | 0.8582 | 0.6704 | **0.9514** |
| Tool clasper | 0.2033 | 0.2095 | 0.1884 | **0.7586** |
| Tool wrist | 0.1041 | 0.0926 | 0.1976 | **0.8026** |
| Tool shaft | 0.5225 | 0.5279 | 0.4735 | **0.9184** |
| Suturing needle | 0.1929 | 0.2027 | 0.1279 | **0.5797** |
| Thread | 0.2552 | 0.2561 | 0.0665 | **0.5130** |
| Suction tool | 0.0244 | 0.0257 | 0.0362 | **0.7175** |
| Needle holder | 0.0574 | 0.0428 | 0.2313 | **0.7738** |
| Clamps | 0.1005 | 0.1016 | 0.0232 | **0.4351** |
| Catheter | 0.5525 | 0.5129 | 0.3634 | **0.8212** |

### Ablation Study

**Design choice 1: Loss function (SegFormer AdamW)**

| Configuration | CE weight | Dice weight | mIoU | Dice | Pixel Acc |
|---|---|---|---|---|---|
| CE only | 1.0 | 0.0 | 0.6455 | 0.7623 | 0.9293 |
| CE + Dice (balanced) | 0.5 | 0.5 | **0.7271** | **0.8314** | **0.9552** |

**Design choice 2: Backbone fine-tuning (SegFormer AdamW)**

| Configuration | mIoU | Dice | Pixel Acc | Params trained |
|---|---|---|---|---|
| Frozen encoder | 0.4855 | 0.6179 | 0.8678 | Decode head only |
| Full fine-tuning | **0.7271** | **0.8314** | **0.9552** | All 27M params |

---

### Key Findings

**1. AdamW dramatically outperforms Adam for SegFormer fine-tuning.**
SegFormer with AdamW achieves mIoU of 0.7271 vs 0.2378 with Adam — a 20.6 percentage point improvement. This is the single largest performance gap in the benchmark and confirms that AdamW's decoupled weight decay is significantly more effective than Adam's coupled regularization for fine-tuning transformer architectures. The pixel accuracy improvement (95.5% vs 65.1%) further demonstrates that Adam with this learning rate severely underfits on this task.

**2. CNN models (UNet, Attention UNet) plateau well below SegFormer AdamW despite similar pixel accuracy.**
UNet and Attention UNet achieve comparable pixel accuracy (~83%) to each other but achieve only ~0.287 mIoU — less than half of SegFormer AdamW's 0.727. This discrepancy is explained by pixel accuracy being dominated by the background class which all models segment well. mIoU penalizes poor minority class performance equally, exposing the CNN models' weakness on small instruments. Attention UNet and UNet perform nearly identically (0.2830 vs 0.2871 mIoU), suggesting attention gates provide minimal benefit at this scale without a stronger backbone.

**3. Compound CE+Dice loss outperforms CE-only across all metrics.**
Adding Dice loss (CE weight 0.5, Dice weight 0.5) improves mIoU from 0.6455 to 0.7271 (+8.2 points) and Dice score from 0.7623 to 0.8314 (+6.9 points) over CE-only training. The improvement is most pronounced on minority classes — suturing needle IoU improves from 0.4350 to 0.5797 (+14.5 points) and clamps IoU from 0.2379 to 0.4351 (+19.7 points). This confirms that Dice loss's invariance to class imbalance is critical for rare instrument classes that CE-only training underweights despite the inverse-frequency class weights applied to CE.

**4. Full fine-tuning of the SegFormer encoder is essential.**
Freezing the MiT-b2 encoder and training only the decode head reduces mIoU from 0.7271 to 0.4855 — a drop of 24.2 percentage points. The performance gap is consistent across all instrument classes, with the largest drops on suction tool (0.7175 → 0.1619, −55.6 points) and clamps (0.4351 → 0.1407, −29.4 points). This demonstrates that ImageNet pretrained features require substantial adaptation for the surgical imaging domain — the frozen encoder's natural image features are insufficient for distinguishing between visually similar instrument classes in endoscopic video.

**5. Small and rare instruments remain the hardest classes across all models.**
Even the best model (SegFormer AdamW) struggles on clamps (IoU=0.4351) which appear infrequently, and suturing needle (IoU=0.5797) and thread (IoU=0.5130) which are physically small. All CNN-based models score below 0.26 IoU on every instrument class except tool shaft and catheter. The consistent failure on these classes across architectures suggests the bottleneck is data quantity for rare classes rather than model capacity.

**6. SegFormer AdamW achieves strong performance on large, common instruments.**
Tool shaft (IoU=0.9184), tool wrist (IoU=0.8026), and catheter (IoU=0.8212) all exceed 0.80 IoU with SegFormer AdamW — these are the largest and most frequently occurring instrument classes. The hierarchical Mix Transformer encoder's ability to capture both fine-grained local features and global context likely explains its advantage over UNet-based architectures on these classes.

---

### Dataset

| Split | Videos | Frames |
|---|---|---|
| Train | 1–36 | 11,862 |
| Val | 37–40 | 1,181 |
| Test | 41–50 | 3,252 |
| **Total** | **50** | **16,295** |

Split performed at the **video level** to prevent temporal data leakage between frames from the same surgery.

---

## Individual Contributions

Solo project — all work completed individually.

---

## Project Structure

```
.
├── src/
│   ├── preprocess_sar_rarp50.py   # Frame extraction, splits, class weights
│   ├── dataset.py                 # PyTorch Dataset, DataLoader, augmentations
│   ├── models.py                  # UNet, Attention UNet, SegFormer, CompoundLoss
│   ├── train.py                   # Training loop, DDP, scheduling, early stopping
│   ├── evaluate.py                # Curves, metrics, confusion matrix, failures
│   └── evaluate2.py               # Per-class Dice, pixel accuracy, predictions
├── data/
│   └── download_data.sh           # Kaggle download script
├── models/                        # Saved model checkpoints (.pth)
├── notebooks/                     # Jupyter notebooks for exploration
├── videos/                        # Demo and walkthrough videos
├── docs/
│   └── walkthrough_script.md      # Technical walkthrough script
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
└── requirements.txt
```


## Attribution:
Generative AI was utilized to improve visualization of this file.