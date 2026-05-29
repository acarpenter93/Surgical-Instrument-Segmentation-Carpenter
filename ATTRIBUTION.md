# Attribution

## AI Development Tools

### What was AI-generated

- **Code Headers** for all five modules (`preprocess_sar_rarp50.py`, `dataset.py`, `models.py`, `train.py`, `evaluate.py`) — code headers were generated with AI. (Generative AI was used to reformat most files to be more visually appealing -- headers, spacing, etc.)
- **DistributedDataParallel boilerplate** — the DDP setup, `DistributedSampler` integration, and rank helper functions were partially generated (modified my original implementation) with AI assistance to better match PyTorch documentation patterns

### What was modified or required debugging

- **Data Preprocessing** — AI was implemented to help debug preprocess_sar_rarp50.py (which was modified based on my old code for a different project/dataset) and dataset.py as the first implementation was running into errors with (various) mismatching characteristics
- **The `logits.float()` fix** — AI-generated code produced a `RuntimeError: expected scalar type Half but found Float` crash when mixed precision autocast produced float16 logits but class weights were float32. This required diagnosing the type mismatch and adding explicit casting before the loss computation
- **The `dist.broadcast` indentation bug** — the early stopping broadcast was incorrectly nested inside the wrong `if` block, causing DDP processes to desync and hang. Required manual debugging of the control flow
- **Video-level split logic** — the split video naming convention (`video_11_1`, `video_11_2`) was not handled in the initial version and required a custom `parse_video_num` function to correctly assign split parts to the same train/val bucket
- **`INTER_NEAREST` for mask resizing** — initial version used default interpolation for masks, which blended class label values. Required identifying this as the source of invalid label values and switching to nearest neighbor interpolation
- **Multi-GPU parallelism issues** — running parallel training processes with `&` and `wait` in shell scripts caused process stacking when scripts were restarted. Required abandoning the parallel shell approach and switching to sequential `torchrun` calls
- **Mixed GPU hardware** — discovered the cluster had mixed GPU types (P100 + RTX 2080 Ti) which causes DDP slowdowns due to speed mismatch. Required using `CUDA_VISIBLE_DEVICES` to restrict to matched hardware. This was also fixed (by myself, not with AI) with the slurm argument when starting my machine/interactive session

### What was written independently

- actual code in src files -- (`preprocess_sar_rarp50.py`, `dataset.py`, `models.py`, `train.py`, `evaluate.py`) (first two files needing AI assistance for debudding/modifying)
- all .txt files 
- overnight.sh (modified from previous project to fit current project)
- All experimental design decisions (split strategy, loss weighting, optimizer choice, ablation configurations)
- Debugging of all runtime errors encountered during actual training runs
- Analysis and interpretation of results

---

## External Libraries

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.1.0 | Core deep learning framework |
| torchvision | ≥0.16.0 | Image utilities |
| HuggingFace transformers | ≥4.40.0 | SegFormer pretrained model |
| albumentations | ≥1.4.0 | Image augmentation pipeline |
| OpenCV | ≥4.9.0 | Video frame extraction, image I/O |
| NumPy | ≥1.26.0 | Numerical computation, confusion matrix |
| matplotlib | ≥3.8.0 | All visualization and plotting |
| tqdm | ≥4.66.0 | Progress bars during preprocessing |

---

## Pretrained Models

- **SegFormer mit-b2** (`nvidia/mit-b2`) — Mix Transformer encoder pretrained on ImageNet-1K, provided by NVIDIA via HuggingFace Model Hub. Used as the backbone for SegFormer fine-tuning. License: Apache 2.0.

---

## Dataset

- **SAR-RARP50** — Surgical Activity Recognition dataset from 50 real Robot-Assisted Radical Prostatectomy operations. Provided under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Accessed via Kaggle mirrors:
  - https://www.kaggle.com/datasets/umarfrq/sar-rarp50-train-set
  - https://www.kaggle.com/datasets/umarfrq/sar-rarp50-test-set

---

## Research Paper

This project reproduces and extends:

> Ameli, S. (2026). *Benchmarking CNN- and Transformer-Based Models for Surgical Instrument Segmentation in Robotic-Assisted Surgery*. arXiv:2604.09151.

Reproduced experimental setup: Adam optimizer, lr=1e-4, batch size 4, 384×384 resolution, 10 epochs, compound CE+Dice loss.

---

## Architecture References

- **UNet:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- **Attention UNet:** Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *MIDL*.
- **SegFormer:** Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*.
- **Dice Loss:** Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV*.
- **AdamW:** Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
