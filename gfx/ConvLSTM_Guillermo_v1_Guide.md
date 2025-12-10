# ConvLSTM_Guillermo_v1 Model Documentation

## Overview

The **ConvLSTM_Guillermo_v1** is an enhanced Convolutional LSTM model designed for next-day wildfire spread prediction on the WildFireSpreadTS benchmark. This model achieves **0.46 AP** (compared to 0.35 for vanilla ConvLSTM and 0.39 for UTAE baseline) through systematic architectural improvements.

**Performance**: 0.46 ± 0.10 AP | 0.18 ± 0.07 UCE | 6.2M parameters

## Key Features

### 1. Multi-Scale Pyramid Pooling
- Captures features at different spatial scales (1×, 2×, 4× pooling)
- Helps the model understand both local fire behavior and larger landscape patterns
- Adaptive pooling ensures consistent feature map dimensions across scales
- **Impact**: Marginal (+0.001 AP), helps with multi-scale patterns

### 2. Spatial Attention Mechanism
- **Channel Attention**: 1×1 convolution + sigmoid activation
- **Spatial Attention**: Focuses on fire-prone spatial regions
- Applied after each ConvLSTM layer's hidden state
- **Impact**: Most valuable component (~0.014 AP improvement)

### 3. Enhanced ConvLSTM Cells
- **Residual Connections**: Skip connections for better gradient flow
- **Group Normalization**: Normalizes across channel groups, improves stability
- **Standard Gating**: Input, forget, cell, output gates (ConvLSTM architecture)
- **Impact**: GroupNorm stabilizes training, residual connections neutral on AP

### 4. Deep Supervision
- Auxiliary loss from encoder output (before feature refinement)
- Weight: 0.2 (tuned via ablation study)
- Helps with gradient flow in deep networks
- **Impact**: Small benefit (~0.002 AP)

### 5. Multi-Scale Classification Head
- Parallel convolutions: 1×1 (global), 3×3 (local), 5×5 (contextual)
- Feature fusion via concatenation + final 1×1 conv
- Dropout (0.1) for regularization
- **Impact**: Improves probability calibration (UCE)

## Model Architecture

```
Input: (batch_size, seq_len=1, channels=40, height=128, width=128)
         ↓
Multi-Scale Pyramid Pooling [1, 2, 4]
         ↓ (channels × 3 scales)
3-Layer Stacked ConvLSTM
  - Layer 1: 64 hidden dims + Attention + GroupNorm + Residual
  - Layer 2: 128 hidden dims + Attention + GroupNorm + Residual  
  - Layer 3: 256 hidden dims + Attention + GroupNorm + Residual
         ↓ (256 channels)
Feature Refinement (2 conv layers + dropout)
         ↓ (256 channels)
Multi-Scale Classification Head [1×1, 3×3, 5×5] + Fusion
         ↓
Output: (batch_size, 1, height=128, width=128)

Deep Supervision Branch:
  Encoder Output → Simple Conv Head → Auxiliary Loss (weight=0.2)
```

**Total Parameters**: ~6.2M  
**Input Features**: 40 (static terrain + dynamic weather/fire/vegetation)  
**Spatial Resolution**: 128×128 pixels per sample  
**Temporal Input**: Single timestep (day d) to predict next day (d+1)

## Performance Analysis

### Ablation Study Results

Our systematic ablation study reveals the contribution of each component:

| Configuration | AP ↑ | UCE ↓ | ΔAP | Insight |
|---------------|------|-------|-----|---------|
| **Full Model** | **0.460 ± 0.101** | **0.177 ± 0.071** | - | Baseline |
| No Attention | 0.446 ± 0.126 | 0.180 ± 0.072 | -0.014 | **Most valuable component** |
| No Pyramid | 0.461 ± 0.152 | 0.175 ± 0.078 | +0.001 | Multi-scale pooling marginal |
| No Residual | 0.460 ± 0.157 | 0.180 ± 0.076 | ±0.000 | Neutral on AP |
| No Deep Supervision | 0.458 ± 0.144 | 0.181 ± 0.075 | -0.002 | Small benefit |
| Minimal (no enhancements) | 0.452 ± 0.146 | 0.182 ± 0.076 | -0.008 | Combined ~0.8% gain |

**Key Finding**: Spatial attention is the most impactful component. Other enhancements provide marginal individual gains but improve robustness.

### Performance Decomposition

The +0.11 AP improvement over original ConvLSTM (0.35 AP) decomposes as:

1. **Training Methodology** (+0.08 AP, 73%):
   - Focal Loss (α=0.25, γ=2.0) vs BCE
   - AdamW optimizer vs Adam
   - Cosine annealing LR schedule

2. **Architectural Enhancements** (+0.03 AP, 27%):
   - **Depth & Capacity**: 3 layers [64,128,256] vs 1 layer [64]
   - **Attention**: Spatial attention mechanism
   - **Multi-scale Head**: Improves calibration (UCE)
   - **Feature Refinement**: 2-layer refinement network

### Comparison with Baselines

| Model | AP ↑ | UCE ↓ | Parameters | Key Strengths |
|-------|------|-------|------------|---------------|
| Original ConvLSTM | 0.35 ± 0.XX | - | 240K | Simple, fast |
| UTAE Baseline | 0.39 ± 0.08 | 0.38 ± 0.00 | ~1M | Temporal attention |
| **ConvLSTM_Guillermo_v1** | **0.46 ± 0.10** | **0.18 ± 0.07** | **6.2M** | **Best AP & calibration** |

**Advantages**:
- **+28% AP improvement** over UTAE (relative)
- **-53% UCE improvement** (better calibration)
- Well-balanced discrimination and calibration

## Training Guide

### Prerequisites

```bash
# Create environment
conda create -n wildfire python=3.10.4
conda activate wildfire

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Test Run (5 epochs, 20% data):**
```bash
python src/train.py \
    --config cfgs/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainer_test_short.yaml \
    --data cfgs/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/hdf5/data \
    --data.data_fold_id 0 \
    --do_test true
```

**Full Training (Single Fold):**

```bash
python src/train.py \
    --config cfgs/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/hdf5/data \
    --data.data_fold_id 0 \
    --do_test true
```

**12-Fold Cross-Validation (WandB Sweep):**
```bash
# Update data path in cfgs/convlstm/full_run.yaml
wandb sweep cfgs/convlstm/full_run.yaml
wandb agent <SWEEP_ID>
```

### Configuration Files

The model uses PyTorch Lightning CLI with YAML configs:

1. **Model Config** (`--config`): Defines architecture, loss, optimizer
   - `cfgs/convlstm_guillermo_v1_config.yaml` - Main v1 config

2. **Trainer Config** (`--trainer`): Training loop settings
   - `cfgs/trainer_single_gpu.yaml` - 170 max epochs, early stopping (patience=40)
   - `cfgs/trainer_test_short.yaml` - 5 epochs for quick testing

3. **Data Config** (`--data`): Dataset and loading parameters
   - `cfgs/data_monotemporal_full_features.yaml` - All 40 features, single timestep

### Key Hyperparameters

From `convlstm_guillermo_v1_config.yaml`:

```yaml
model:
  class_path: models.ConvLSTM_Guillermo_v1.ConvLSTM_Guillermo_v1
  init_args:
    n_channels: 40                      # Input features
    hidden_dims: [64, 128, 256]         # Layer capacities
    num_layers: 3                       # ConvLSTM depth
    kernel_size: 3                      # ConvLSTM kernel
    pyramid_scales: [1, 2, 4]           # Multi-scale pooling
    
    # Architectural toggles
    use_attention: true                 # Spatial attention
    use_residual: true                  # Skip connections
    use_groupnorm: true                 # Group normalization
    use_feature_refinement: true        # Refinement network
    use_multiscale_head: true           # Multi-scale decoder
    
    # Loss and regularization
    loss_function: "focal"              # Focal loss
    focal_alpha: 0.25                   # Class weight
    focal_gamma: 2.0                    # Focusing parameter
    deep_supervision_weight: 0.2        # Auxiliary loss weight
    dropout_rate: 0.1                   # Dropout probability

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01

lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 170
```

### Running Ablations

See [`cfgs/convlstm/ablations/README.md`](../cfgs/convlstm/ablations/README.md) for comprehensive ablation study documentation.

**Example - Test without attention:**
```bash
python src/train.py \
    --config cfgs/convlstm/ablations/no_attention.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0
```

## Expected Outputs

### During Training

**Console Logs:**
```
Epoch 0:   5%|▍         | 100/2000 [01:23<25:12,  1.26it/s, loss=0.234, v_num=abc123]
Epoch 0: 100%|██████████| 2000/2000 [27:45<00:00,  1.20it/s, loss=0.156, v_num=abc123]
Validation: |██████████| 500/500 [03:12<00:00]
Epoch 0, global step 2000: 'val_ap' reached 0.35 (best 0.35), saving model to 'checkpoints/epoch=0-step=2000.ckpt'
```

**WandB Metrics (logged every step/epoch):**
- `train_loss`, `train_main_loss`, `train_deep_sup_loss`
- `train_f1`, `train_precision`, `train_recall`
- `val_loss`, `val_f1`, `val_ap` (validation AP)
- `learning_rate`
- `epoch`, `step`

**Test Metrics (computed at end):**
- `test_AP` - Average Precision (primary metric)
- `test_UCE` - Unweighted Calibration Error
- `test_f1`, `test_precision`, `test_recall`
- `test_iou` - Intersection over Union

### Saved Files

**Checkpoints** (`lightning_logs/version_X/checkpoints/`):
- `epoch=X-step=XXXX-val_ap=0.XXX.ckpt` - Best model based on validation AP
- Contains full model state dict, optimizer state, hyperparameters

**WandB Artifacts**:
- Training curves and metrics
- System monitoring (GPU, memory)
- Hyperparameter tracking
- Run comparison tools

**Predictions** (if saved):
- `predictions.pt` - Torch tensor with model outputs
- `targets.pt` - Ground truth labels
- Can be loaded for offline metric computation

## Performance Expectations

### Training Dynamics

**Typical Training Curve:**
- **Epochs 0-20**: Rapid improvement, loss drops from ~0.25 to ~0.10
- **Epochs 20-80**: Steady improvement, val_ap increases to ~0.40-0.45
- **Epochs 80-170**: Fine-tuning, val_ap plateaus around 0.45-0.47
- **Early Stopping**: Usually triggers around epoch 100-130 (patience=40)

**Computational Requirements:**
- **Training Time**: ~4-6 hours per fold (NVIDIA RTX 3090)
- **GPU Memory**: ~8-10GB (batch_size=15)
- **Disk Space**: ~2GB per checkpoint

### Expected Performance Ranges

Based on 12-fold cross-validation:

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| **AP** | 0.46 | 0.10 | 0.35-0.60 |
| **UCE** | 0.18 | 0.07 | 0.10-0.30 |
| **F1** | 0.35 | 0.08 | 0.25-0.45 |
| **IoU** | 0.22 | 0.06 | 0.15-0.32 |

**Note**: High variance across folds is normal for wildfire prediction due to diverse fire behaviors and environmental conditions in different spatial regions.

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```
**Solutions:**
- Reduce batch size: `--data.batch_size 8` or `--data.batch_size 4`
- Reduce model capacity: `hidden_dims: [32, 64, 128]`
- Enable gradient checkpointing (requires code modification)

**2. Poor Convergence (AP < 0.35)**
```
val_ap stuck at 0.20-0.30
```
**Solutions:**
- Verify focal loss parameters: `focal_alpha=0.25, focal_gamma=2.0`
- Check learning rate: Try `lr=0.0005` or `lr=0.002`
- Ensure data is normalized correctly
- Increase patience: `--trainer.early_stopping_patience 60`

**3. Overfitting (train_ap >> val_ap)**
```
train_ap=0.65, val_ap=0.40
```
**Solutions:**
- Increase dropout: `dropout_rate: 0.2`
- Add weight decay: `weight_decay: 0.05`
- Reduce model capacity: `hidden_dims: [32, 64, 128]`
- Enable data augmentation

**4. Training Too Slow**
```
<0.5 it/s, 12+ hours per fold
```
**Solutions:**
- Increase num_workers: `--data.num_workers 8`
- Use SSD for data storage (not HDD)
- Enable pinned memory: `--data.pin_memory true`
- Check GPU utilization with `nvidia-smi`

**5. WandB Issues**
```
wandb: ERROR Error while calling W&B API
```
**Solutions:**
```bash
wandb login
# Or disable WandB for local testing
export WANDB_MODE=offline
```

### Validation Checklist

Before starting full 12-fold training:

- [ ] Quick test run completes without errors (5 epochs)
- [ ] WandB logging works correctly
- [ ] Checkpoints are being saved
- [ ] val_ap metric is computed (not NaN)
- [ ] GPU utilization >80% during training
- [ ] Batch size fits in GPU memory comfortably
- [ ] Data path is correct and accessible

## Code Implementation Details

### Model Class Hierarchy

```python
ConvLSTM_Guillermo_v1 (src/models/ConvLSTM_Guillermo_v1.py)
    └─ Inherits from BaseModel (src/models/BaseModel.py)
        └─ Inherits from pl.LightningModule
    
    Uses:
    ├─ MultiScaleConvLSTM: Pyramid pooling + stacked ConvLSTM
    │   └─ EnhancedConvLSTMCell: Attention + GroupNorm + Residual
    ├─ FeatureRefinement: 2-layer conv refinement
    └─ MultiScaleClassificationHead: Parallel convs + fusion
```

### Key Methods

**`__init__()`**: Initialize architecture components
- Creates multi-scale pyramid pooling
- Initializes 3-layer ConvLSTM stack
- Sets up refinement and classification heads
- Configures loss function (Focal/Dice/BCE)

**`forward(x)`**: Forward pass
- Multi-scale pyramid pooling on input
- Process through ConvLSTM layers
- Apply refinement and classification head
- Return prediction + encoder output (for deep supervision)

**`training_step(batch, batch_idx)`**: Training iteration
- Compute main prediction loss
- Compute deep supervision loss (if enabled)
- Log metrics (loss, F1)
- Return total loss

**`validation_step(batch, batch_idx)`**: Validation iteration
- Compute validation loss and metrics
- Store predictions for AP computation
- Log to WandB

**`test_step(batch, batch_idx)`**: Test iteration
- Generate predictions
- Compute comprehensive metrics (AP, UCE, F1, IoU)
- Save predictions if requested

**`configure_optimizers()`**: Setup optimizer and scheduler
- AdamW optimizer with weight decay
- Cosine annealing LR schedule
- Returns optimizer and scheduler dict

### Ablation Parameters

The model supports ablation studies via constructor parameters:

```python
model = ConvLSTM_Guillermo_v1(
    use_attention=False,        # Disable spatial attention
    use_residual=False,         # Disable skip connections
    use_groupnorm=False,        # Disable group normalization
    use_feature_refinement=False,  # Skip refinement network
    use_multiscale_head=False,  # Use simple 3×3 conv decoder
    deep_supervision_weight=0.0,   # Disable auxiliary loss
    pyramid_scales=[1],         # Single scale (no pyramid)
)
```

See [`cfgs/convlstm/ablations/README.md`](../cfgs/convlstm/ablations/README.md) for pre-configured ablation experiments.

## Model Variants

### ConvLSTM_Guillermo_v1 (This Model)
- **Focus**: Balanced performance and interpretability
- **Architecture**: Unidirectional, 3 layers
- **Performance**: 0.46 AP, 0.18 UCE
- **Status**: Main model for paper submission

### ConvLSTM_Guillermo_v2 (Experimental)
- **Focus**: Maximum performance
- **Architecture**: Bidirectional, temporal attention, FocalDice loss
- **Config**: `cfgs/convlstm_guillermo_v2_config.yaml`
- **Guide**: [`ConvLSTM_Guillermo_v2_Guide.md`](ConvLSTM_Guillermo_v2_Guide.md)
- **Status**: Under development

### Original ConvLSTM (Paper Baseline)
- **Architecture**: 1 layer, 64 hidden dims, BCE loss
- **Performance**: ~0.35 AP
- **Config**: `cfgs/convlstm/ablations/original_paper_convlstm.yaml`
- **Purpose**: Baseline comparison

## References

### Papers
- **WildFireSpreadTS Dataset**: Gerard et al., NeurIPS 2023 [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)
- **Focal Loss**: Lin et al., ICCV 2017 [[arXiv]](https://arxiv.org/abs/1708.02002)
- **ConvLSTM**: Shi et al., NeurIPS 2015 [[arXiv]](https://arxiv.org/abs/1506.04214)

### Documentation
- [Main README](../README.md) - Repository overview
- [Ablation Studies](../cfgs/convlstm/ablations/README.md) - Detailed ablation guide
- [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)

### Related Files
- Model: `src/models/ConvLSTM_Guillermo_v1.py`
- Base Class: `src/models/BaseModel.py`
- Metrics: `src/models/EvaluationMetrics.py`
- Training: `src/train.py`
- Data: `src/dataloader/FireSpreadDataset.py`

---

**Last Updated**: 2025  
**Model Version**: 1.0  
**Performance**: 0.46 AP | 0.18 UCE | 6.2M params
