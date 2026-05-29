# Setup Instructions
The exact scripts/path I followed given my (temporary) file setup can be found in InstrumentSegmentationSurgery/docs/stepbystep.txt
Go to this location if further, more in-depth instructions are needed.
(In this setup, all original (src, data, etc.--not produced by code) files were found directly in InstrumentSegmentationSurgery)
I ran this project using Duke's Open OnCommand Jupyte Lab Interactive Sessions, with the specifications:

Environment setup: 
/# Restore module environment to avoid conflicts 
module restore 
/# Load required modules 
module load miniconda/23.9.0
partition: compsci-gpu
Number of GPUs: 0
CPU RAM: 50G
Number of cores: 4
Number of hours: 5
Extra slurm arguments to pass: slurm argument: --gres=gpu:p100:2

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on Tesla P100 12GB)
- ~100GB disk space for raw dataset
- ~20GB disk space for processed frames

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd InstrumentSegmentationSurgery
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with opencv on a headless server:
```bash
pip install opencv-python-headless
```

---

## 4. Download the Dataset

You need a Kaggle account and API key. Set up Kaggle credentials:
```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place your kaggle.json API key file at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download both dataset splits:
```bash
mkdir -p data
cd data

kaggle datasets download -d umarfrq/sar-rarp50-train-set
unzip sar-rarp50-train-set.zip -d train-set

kaggle datasets download -d umarfrq/sar-rarp50-test-set
unzip sar-rarp50-test-set.zip -d test-set

cd ..
```

Expected structure after unzipping:
```
data/
├── train-set/
│   ├── video_01/
│   │   ├── video_left.avi
│   │   └── segmentation/
│   │       ├── 000000000.png
│   │       └── ...
│   ├── video_11_1/
│   ├── video_11_2/
│   └── ... (videos 1-40)
└── test-set/
    ├── video_41/
    └── ... (videos 41-50)
```

---

## 5. Preprocess the Dataset

Extracts frames from videos at 1Hz, resizes to 384×384, builds
train/val/test manifests, and computes class weights:

```bash
python preprocess_sar_rarp50.py \
    --train_root ./data/train-set \
    --test_root  ./data/test-set \
    --output_dir ./processed \
    --verify
```

This takes 20-40 minutes depending on your disk speed.
Output: `processed/train_manifest.json`, `val_manifest.json`,
`test_manifest.json`, `dataset_stats.json`, and extracted frames.

---

## 6. Verify Dataset Loads Correctly

```bash
python dataset.py --processed_dir ./processed --visualize
```

This saves a `sample_train_batch.png` showing augmented frames and masks.

---

## 7. Train Models

**Single GPU:**
```bash
python train.py \
    --model segformer \
    --optimizer adamw \
    --lr 1e-4 \
    --processed_dir ./processed \
    --output_dir ./runs \
    --batch_size 4 --img_size 384 --epochs 10
```

**Multi-GPU (recommended):**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --model segformer \
    --optimizer adamw \
    --lr 1e-4 \
    --processed_dir ./processed \
    --output_dir ./runs \
    --num_workers 2 --batch_size 4 --img_size 384 --epochs 10
```

**Available models:** `segformer`, `unet`, `attention_unet`  
**Available optimizers:** `adam`, `adamw`, `sgd`

---

## 8. Evaluate

```bash
# Training curves
python evaluate.py --mode curves \
    --run_dir runs/<run_folder>

# Full test set evaluation
python evaluate.py --mode eval \
    --run_dir runs/<run_folder> \
    --processed_dir ./processed \
    --batch_size 4 --num_workers 2

# Compare multiple runs
python evaluate.py --mode compare \
    --run_dirs runs/run1 runs/run2 runs/run3 \
    --processed_dir ./processed \
    --output_dir ./comparison \
    --batch_size 4 --num_workers 2

# Failure case visualization
python evaluate.py --mode failures \
    --run_dir runs/<run_folder> \
    --processed_dir ./processed \
    --batch_size 4 --num_workers 2

# Per-class Dice + pixel accuracy charts
python evaluate2.py --mode per_class \
    --run_dirs runs/run1 runs/run2 runs/run3 \
    --output_dir ./comparison2

# Prediction visualization (input + GT + prediction)
python evaluate2.py --mode predict \
    --run_dirs runs/run1 runs/run2 runs/run3 \
    --processed_dir ./processed \
    --output_dir ./comparison2 \
    --n_samples 5
```

---

## Notes 

- Preprocessing must be run before training — training reads from the manifest JSON files
- The first SegFormer run downloads ~100MB of pretrained weights from HuggingFace automatically
- All outputs (checkpoints, curves, plots) are saved to the `runs/` directory under a timestamped folder
- If you do not have a GPU, set `--num_workers 0` and expect significantly slower training

