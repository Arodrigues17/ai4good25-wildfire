# Enhanced ConvLSTM for Wildfire Spread Prediction

This repository contains the implementation of **Enhanced ConvLSTM** models for next-day wildfire spread prediction, developed as part of the WildFireSpreadTS benchmark evaluation.

If you would like to run the TransformerCA variant of the repository please switch branches to `feature/TransformerCA`, UTAE_physics for physics loss function applied to UTAE, and transformer_ViT to see archived transformers model.

## 📋 Overview

Wildfire spread prediction is critical for emergency management and resource allocation. This work presents an enhanced ConvLSTM architecture that achieves **0.46 AP** (compared to 0.39 for UTAE baseline and 0.35 for vanilla ConvLSTM) through systematic architectural improvements.

### Key Contributions

1. **Enhanced ConvLSTM Architecture** with:
   - Multi-scale spatial pyramid pooling
   - Channel-spatial attention mechanisms
   - Stacked recurrent layers (3 layers: 64→128→256 hidden dims)
   - Group normalization and residual connections
   - Multi-scale classification head
   - Deep supervision

2. **Comprehensive Ablation Study** demonstrating:
   - Impact of model depth and capacity
   - Effect of architectural components (attention, pyramid pooling, etc.)
   - Importance of training methodology (Focal loss, AdamW optimizer)

3. **Well-Calibrated Predictions**: UCE = 0.18 (compared to 0.38 for UTAE baseline)

![](gfx/wildfirespreadts_overview.png)

**The WildFireSpreadTS** benchmark is a machine learning-ready dataset to evaluate next-day spread prediction models. Given a binary map of the active fire locations during day d, the task is to predict where the fire will be active on day d+1. Models have access to a 5-day history of remote sensing observations and weather conditions, static topographic information, and weather forecasts.

Read the [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf) to learn more about the dataset.

## 🚀 Quick Start

### 1. Environment Setup


```bash
git clone https://github.com/Arodrigues17/ai4good25-wildfire.git
cd ai4good25-wildfire

# Create Python 3.10 environment
conda create -n wildfire python=3.10.4
conda activate wildfire
pip install -r requirements.txt
```

### 2. Data Preparation

Download and process the WildFireSpreadTS dataset:

```bash
# Download dataset (~30GB compressed)
wget https://zenodo.org/api/records/8006177/files-archive

# Unzip (twice) and convert to HDF5 format
python src/preprocess/CreateHDF5Dataset.py \
    --data_dir /path/to/raw/data \
    --target_dir /path/to/hdf5/output
```

### 3. Train Enhanced ConvLSTM

**Quick Test (5 epochs):**
```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainers/trainer_test_short.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/hdf5/data \
    --data.data_fold_id 0 \
    --do_test true
```

**Full Training (Single Fold):**
```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/hdf5/data \
    --data.data_fold_id 0 \
    --do_test true
```

**12-Fold Cross-Validation:**
```bash
# Using WandB sweeps for automated execution
wandb sweep cfgs/models/convlstm_v1/wandb_12fold_cv.yaml
wandb agent <SWEEP_ID>
```

## 📊 Results

| Model | AP ↑ | UCE ↓ | Parameters |
|-------|------|-------|------------|
| Original ConvLSTM (paper) | 0.35 ± 0.XX | 0.XX ± 0.XX | 240K |
| UTAE Baseline | 0.39 ± 0.08 | 0.38 ± 0.00 | ~1M |
| **ConvLSTM_Guillermo_v1 (Ours)** | **0.46 ± 0.10** | **0.18 ± 0.07** | **6.2M** |

*Results averaged over 12-fold cross-validation on WildFireSpreadTS benchmark*

### Performance Analysis

Our enhanced ConvLSTM achieves substantial improvements over baselines:
- **+0.11 AP** over UTAE baseline (+28% relative improvement)
- **+0.11 AP** over original ConvLSTM (+31% relative improvement)  
- **-0.20 UCE** better calibration than UTAE (53% reduction in calibration error)

**Key Finding**: The performance gain decomposes as:
- **+0.08 AP** from training methodology (Focal loss + AdamW)
- **+0.03 AP** from architectural enhancements (depth, capacity, attention)

## 🔬 Ablation Studies

We provide comprehensive ablation experiments to understand component contributions. See [`cfgs/models/convlstm_v1/ablations/README.md`](cfgs/models/convlstm_v1/ablations/README.md) for detailed documentation.

### Stage 1: Component Ablations

Tests individual architectural components (pyramid pooling, attention, residual connections, deep supervision).

| Ablation | AP ↑ | UCE ↓ | Description |
|----------|------|-------|-------------|
| **Baseline (Full)** | **0.460 ± 0.101** | **0.177 ± 0.071** | All components enabled |
| No Pyramid | 0.461 ± 0.152 | 0.175 ± 0.078 | Single scale instead of [1,2,4] |
| **No Attention** | **0.446 ± 0.126** | **0.180 ± 0.072** | **Largest drop: -0.014 AP** |
| No Residual | 0.460 ± 0.157 | 0.180 ± 0.076 | Skip residual connections |
| No Deep Supervision | 0.458 ± 0.144 | 0.181 ± 0.075 | Encoder-only loss |
| Minimal | 0.452 ± 0.146 | 0.182 ± 0.076 | All enhancements disabled |

**Key Findings**:
- Spatial attention provides most value (~0.014 AP improvement)
- Other components have marginal individual impact (<0.01 AP each)
- Combined effect is ~0.008 AP vs minimal configuration

### Stage 2: Architecture Ablations

Tests fundamental design choices (depth, capacity, normalization, classification head).

**Available Configurations** (see ablations README for details):
- `stage2_single_layer.yaml` - Test 1 layer vs 3 layers
- `stage2_two_layers.yaml` - Test intermediate depth
- `stage2_small_capacity.yaml` - Half dimensions [32,64,128]
- `stage2_no_groupnorm.yaml` - Remove normalization
- `stage2_simple_head.yaml` - Single 3×3 conv decoder
- `stage2_no_refinement.yaml` - Remove feature refinement
- `stage2_paper_baseline.yaml` - Match original ConvLSTM architecture

## 📂 Repository Structure

```
├── cfgs/                                    # Configuration files
│   ├── models/
│   │   ├── convlstm_v1/                    # ✅ Official v1 model
│   │   │   ├── convlstm_guillermo_v1_config.yaml
│   │   │   ├── wandb_12fold_cv.yaml        # 12-fold CV sweep
│   │   │   └── ablations/                   # Ablation study configs
│   │   │       ├── README.md                # **Detailed ablation guide**
│   │   │       ├── baseline.yaml
│   │   │       ├── stage2_*.yaml
│   │   │       └── wandb_*.yaml
│   │   └── convlstm_v2_experimental/        # ⚠️ Experimental v2
│   │       └── convlstm_guillermo_v2_config.yaml
│   ├── trainers/                            # Training configurations
│   │   ├── trainer_single_gpu.yaml          # Standard (170 epochs)
│   │   ├── trainer_test_short.yaml          # Quick test (5 epochs)
│   │   └── trainer_original_paper.yaml      # Baseline comparison
│   ├── data/                                # Dataset configurations
│   │   ├── data_monotemporal_full_features.yaml  # ✅ Recommended
│   │   └── data_multitemporal_*.yaml
│   ├── baselines/                           # Baseline comparisons
│   ├── README.md                            # **Config usage guide**
│   └── [unet/, UTAE/, LogisticRegression/]  # Other baselines
├── src/
│   ├── models/
│   │   ├── ConvLSTM_Guillermo_v1.py        # **Enhanced ConvLSTM**
│   │   ├── ConvLSTM_Guillermo_v2.py        # V2 (bidirectional variant)
│   │   ├── ConvLSTMLightning.py            # Original paper baseline
│   │   ├── BaseModel.py                     # Base training logic
│   │   ├── EvaluationMetrics.py            # AP and UCE computation
│   │   └── ...
│   ├── dataloader/                          # Data loading pipeline
│   │   ├── FireSpreadDataModule.py         # Lightning DataModule
│   │   ├── FireSpreadDataset.py            # PyTorch Dataset
│   │   └── utils.py
│   ├── preprocess/
│   │   └── CreateHDF5Dataset.py            # Convert raw data to HDF5
│   └── train.py                             # Main training script
├── gfx/
│   └── wildfirespreadts_overview.png       # Dataset visualization
└── README.md                                # This file
```

## 🏗️ Model Architecture

### ConvLSTM_Guillermo_v1 (Official Released Model)

> **Note**: This is the final, validated model for the paper submission. ConvLSTM_Guillermo_v2 exists as experimental development but is not part of the official release.

**Architecture Overview:**

1. **Input Processing**: Multi-scale pyramid pooling at scales [1, 2, 4]
   - Captures patterns at different spatial resolutions
   - Pooled features concatenated along channel dimension

2. **Temporal Encoder**: 3 stacked ConvLSTM layers with progressive capacity
   - Layer 1: 64 hidden dimensions
   - Layer 2: 128 hidden dimensions (doubles capacity)
   - Layer 3: 256 hidden dimensions (final representation)
   - Total temporal modeling capacity: 6.2M parameters

3. **Enhanced ConvLSTM Cells**:
   - **Group Normalization**: Stabilizes training, normalizes across channel groups
   - **Spatial Attention**: Channel-wise attention gates (1×1 conv + sigmoid)
   - **Residual Connections**: Skip connections for gradient flow

4. **Feature Refinement**: 2-layer convolutional network
   - Refines encoder output before classification
   - Includes dropout for regularization

5. **Multi-Scale Classification Head**: Parallel convolutions
   - 1×1 conv (global patterns)
   - 3×3 conv (local patterns)
   - 5×5 conv (contextual patterns)
   - Outputs fused for final prediction

6. **Deep Supervision**: Auxiliary loss from encoder output
   - Provides learning signal directly to encoder
   - Weight: 0.2 (tuned via ablation)

**Training Configuration:**
- **Loss Function**: Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **LR Schedule**: Cosine annealing with warmup
- **Early Stopping**: Patience=40 epochs on validation AP
- **Batch Size**: 15 (fits in 10GB VRAM)

**Total Parameters**: ~6.2M  
**Inference Time**: ~25ms per sample (NVIDIA RTX 3090)

See `src/models/ConvLSTM_Guillermo_v1.py` for full implementation details.

## 📖 Configuration Guide

> **📚 Complete Guide**: See [`cfgs/README.md`](cfgs/README.md) for comprehensive config documentation.

### Model Configurations

- **`models/convlstm_v1/convlstm_guillermo_v1_config.yaml`** ✅ - Full enhanced model (recommended)
  - 3 layers, hidden_dims=[64,128,256]
  - All enhancements enabled
  - Focal loss, AdamW optimizer

- **`models/convlstm_v2_experimental/convlstm_guillermo_v2_config.yaml`** ⚠️ - Experimental variant
  - Bidirectional ConvLSTM, FocalDice loss
  - Not fully validated, use for research only

### Trainer Configurations

- **`trainers/trainer_single_gpu.yaml`** - Standard training
  - Max 170 epochs, early stopping patience=40
  - Monitors val_ap

- **`trainers/trainer_test_short.yaml`** - Quick testing
  - Max 5 epochs, 20% of data
  - For debugging/prototyping

- **`trainers/trainer_original_paper.yaml`** - Baseline comparison
  - Monitors val_f1 (for ConvLSTMLightning compatibility)

### Data Configurations

- **`data/data_monotemporal_full_features.yaml`** ✅ - All 40 input features
  - Static: elevation, slope, aspect, land cover, etc.
  - Dynamic: NDVI, burned area, weather, etc.
  - Single timestep input

- **`data/data_multitemporal_full_features.yaml`** - Multi-temporal setup
  - 5-day history of observations
  - For UTAE and temporal models

### Ablation Configurations

All ablation configs are in `cfgs/models/convlstm_v1/ablations/`. Each config has:
- **Model config** (e.g., `baseline.yaml`) - For single fold training
- **WandB sweep config** (e.g., `wandb_ablation_baseline.yaml`) - For automated 3-fold runs

See [`cfgs/models/convlstm_v1/ablations/README.md`](cfgs/models/convlstm_v1/ablations/README.md) for complete documentation.

## 🔧 Advanced Usage

### Custom Ablation Study

Create a new config file:

```yaml
# my_ablation.yaml
seed_everything: 0
optimizer: 
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 170
model:
  class_path: models.ConvLSTM_Guillermo_v1.ConvLSTM_Guillermo_v1
  init_args:
    n_channels: 40
    hidden_dims: [64, 128, 256]
    num_layers: 3
    kernel_size: 3
    pyramid_scales: [1, 2, 4]
    use_attention: false  # Your ablation here
    use_residual: true
    use_groupnorm: true
    use_feature_refinement: true
    use_multiscale_head: true
    deep_supervision_weight: 0.2
    loss_function: "focal"
    focal_alpha: 0.25
    focal_gamma: 2.0
do_train: true
do_test: true
```

Run it:
```bash
python src/train.py \
    --config my_ablation.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0
```

### WandB Sweeps for Hyperparameter Search

Create a sweep config:

```yaml
# my_sweep.yaml
program: src/train.py
method: grid
metric:
  name: val_ap
  goal: maximize
parameters:
  config:
    value: cfgs/my_config.yaml
  trainer:
    value: cfgs/trainers/trainer_single_gpu.yaml
  data:
    value: cfgs/data/data_monotemporal_full_features.yaml
  data.batch_size:
    value: 15
  data.data_dir:
    value: /path/to/data
  data.data_fold_id:
    values: [0, 5, 10]  # Run on 3 folds
  do_test:
    value: true
```

Launch sweep:
```bash
wandb sweep my_sweep.yaml
wandb agent <SWEEP_ID>
```

### Evaluate Saved Checkpoints

```python
from src.models.EvaluationMetrics import compute_metrics_and_plots
import torch

# Load saved predictions
predictions = torch.load('predictions.pt')
targets = torch.load('targets.pt')

# Compute metrics
metrics = compute_metrics_and_plots(
    predictions, 
    targets,
    save_dir='./results'
)

print(f"AP: {metrics['ap']:.4f}")
print(f"UCE: {metrics['uce']:.4f}")
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025enhanced,
  title={Enhanced ConvLSTM for Wildfire Spread Prediction},
  author={Your Name and Collaborators},
  journal={Conference/Journal Name},
  year={2025}
}
```

Also cite the WildFireSpreadTS dataset:

```bibtex
@inproceedings{gerard2023wildfirespreadts,
  title={WildFireSpreadTS: A Satellite Image Time Series Benchmark for Wildfire Spread Prediction},
  author={Gerard, Sebastian and Zhao, Yu and Sullivan, Josephine},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WildFireSpreadTS Dataset**: Sebastian Gerard, Yu Zhao, and Josephine Sullivan
- **Original Implementation**: [WildfireSpreadTS GitHub](https://github.com/SebastianGer/WildfireSpreadTS)
- **AI4Good Course**: KTH Royal Institute of Technology

---

**Hardware Requirements**: 150GB SSD, 32GB RAM, 10GB+ GPU VRAM  
**Software Requirements**: Python 3.10, PyTorch 2.0+, PyTorch Lightning, WandB
 

## Credits 

Benchmark and paper authors: Sebastian Gerard, Yu Zhao, and Josephine Sullivan

Original Code: https://github.com/SebastianGer/WildfireSpreadTS

Paper citation:
```
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```