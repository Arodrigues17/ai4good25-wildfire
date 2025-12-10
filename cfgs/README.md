# Configuration Files Guide

This directory contains all configuration files for training models on the WildFireSpreadTS benchmark. Configs are organized by type for easy navigation.

## 📁 Directory Structure

```
cfgs/
├── models/                          # Model architecture configurations
│   ├── convlstm_v1/                # ✅ OFFICIAL RELEASED MODEL (v1)
│   │   ├── convlstm_guillermo_v1_config.yaml    # Main model config
│   │   ├── wandb_12fold_cv.yaml                 # 12-fold cross-validation sweep
│   │   ├── wandb_v1_12fold.yaml                 # Alternative 12-fold sweep
│   │   └── ablations/                           # Component ablation studies
│   │       ├── README.md                        # Detailed ablation guide
│   │       ├── baseline.yaml                    # Full model
│   │       ├── minimal.yaml                     # No enhancements
│   │       ├── no_*.yaml                        # Stage 1 ablations
│   │       ├── stage2_*.yaml                    # Stage 2 ablations
│   │       └── wandb_*.yaml                     # WandB sweep configs
│   │
│   └── convlstm_v2_experimental/   # ⚠️ EXPERIMENTAL (not validated)
│       ├── convlstm_guillermo_v2_config.yaml    # Experimental v2 model
│       └── wandb_v2_12fold.yaml                 # v2 sweep (not recommended)
│
├── trainers/                        # Training loop configurations
│   ├── trainer_single_gpu.yaml      # Standard training (170 epochs, early stopping)
│   ├── trainer_test_short.yaml      # Quick test (5 epochs, 20% data)
│   ├── trainer_original_paper.yaml  # For baseline comparison (monitors val_f1)
│   ├── trainer_v2_full.yaml         # v2 training (200 epochs)
│   └── trainer_v2_test_short.yaml   # v2 quick test
│
├── data/                            # Dataset configurations
│   ├── data_monotemporal_full_features.yaml     # ✅ RECOMMENDED: All 40 features, single timestep
│   ├── data_multitemporal_full_features.yaml    # Multi-temporal (for UTAE, etc.)
│   └── data_multitemporal_full_features_doys.yaml  # With day-of-year encoding
│
├── baselines/                       # Baseline model comparison configs
│   ├── wandb_lr_and_loss_search.yaml           # Hyperparameter search
│   └── wandb_table5.yaml                        # Reproduce paper Table 5
│
├── LogisticRegression/              # Logistic regression baseline
├── unet/                            # UNet baseline
└── UTAE/                            # UTAE baseline
```

## 🚀 Quick Start

### Train ConvLSTM v1 (Official Model)

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

**12-Fold Cross-Validation (WandB Sweep):**
```bash
# Update data path in wandb_12fold_cv.yaml first
wandb sweep cfgs/models/convlstm_v1/wandb_12fold_cv.yaml
wandb agent <SWEEP_ID>
```

### Run Ablation Study

See detailed guide: [`cfgs/models/convlstm_v1/ablations/README.md`](models/convlstm_v1/ablations/README.md)

**Example - Test without attention:**
```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/ablations/no_attention.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/hdf5/data \
    --data.data_fold_id 0 \
    --do_test true
```

## 📋 Configuration Files Explained

### Model Configurations

Located in `cfgs/models/`

**ConvLSTM v1 (Official - Use This!):**
- **`convlstm_v1/convlstm_guillermo_v1_config.yaml`** ✅
  - Final released model
  - 3 layers: [64, 128, 256] hidden dimensions
  - Focal loss, AdamW optimizer
  - All enhancements enabled
  - **Performance**: 0.46 ± 0.10 AP | 0.18 ± 0.07 UCE
  - **Use for**: Official experiments, paper submission

**ConvLSTM v2 (Experimental - Not Recommended):**
- **`convlstm_v2_experimental/convlstm_guillermo_v2_config.yaml`** ⚠️
  - Experimental enhancements (bidirectional, temporal attention)
  - FocalDice loss
  - **Status**: Not fully validated, requires more testing
  - **Use for**: Future research only

### Trainer Configurations

Located in `cfgs/trainers/`

**For ConvLSTM v1:**
- **`trainer_single_gpu.yaml`** - Standard training
  - Max 170 epochs
  - Early stopping: patience=40, monitors `val_ap`
  - Use for full experiments
  
- **`trainer_test_short.yaml`** - Quick testing
  - Max 5 epochs, 20% of data
  - Use for debugging and prototyping
  
- **`trainer_original_paper.yaml`** - Baseline comparison
  - Monitors `val_f1` (for ConvLSTMLightning compatibility)
  - Use with original paper baseline configs

**For ConvLSTM v2 (experimental):**
- **`trainer_v2_full.yaml`** - v2 training (200 epochs)
- **`trainer_v2_test_short.yaml`** - v2 quick test

### Data Configurations

Located in `cfgs/data/`

**Recommended:**
- **`data_monotemporal_full_features.yaml`** ✅
  - All 40 input features
  - Single timestep (day d to predict d+1)
  - Use with ConvLSTM v1
  - Batch size: 15 (fits in 10GB VRAM)

**For temporal models:**
- **`data_multitemporal_full_features.yaml`**
  - 5-day history of observations
  - Use with UTAE and other temporal models
  
- **`data_multitemporal_full_features_doys.yaml`**
  - Same as above + day-of-year encoding

### WandB Sweep Configurations

**For v1 12-fold CV:**
- **`models/convlstm_v1/wandb_12fold_cv.yaml`** - Recommended sweep config
- **`models/convlstm_v1/wandb_v1_12fold.yaml`** - Alternative sweep config

**For ablations:**
- Each ablation has a corresponding `wandb_*.yaml` in `models/convlstm_v1/ablations/`
- See ablations README for details

**For baseline comparisons:**
- **`baselines/wandb_lr_and_loss_search.yaml`** - Hyperparameter search
- **`baselines/wandb_table5.yaml`** - Reproduce paper Table 5

## 🔧 Configuration File Format

All configs use PyTorch Lightning CLI YAML format. Three types of configs are combined:

1. **Model config** (`--config`): Architecture, loss, optimizer
2. **Trainer config** (`--trainer`): Training loop, early stopping, logging
3. **Data config** (`--data`): Dataset, batch size, augmentation

### Example Model Config Structure

```yaml
# Model architecture and training
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
    use_attention: true
    use_residual: true
    use_groupnorm: true
    use_feature_refinement: true
    use_multiscale_head: true
    deep_supervision_weight: 0.2
    loss_function: "focal"
    focal_alpha: 0.25
    focal_gamma: 2.0
    dropout_rate: 0.1

do_train: true
do_test: true
```

### Example Trainer Config

```yaml
# Training loop configuration
max_epochs: 170
accelerator: gpu
devices: 1
precision: 32

callbacks:
  - class_path: pytorch_lightning.callbacks.EarlyStopping
    init_args:
      monitor: val_ap
      patience: 40
      mode: max
  
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      monitor: val_ap
      mode: max
      save_top_k: 1
      filename: "epoch={epoch}-step={step}-val_ap={val_ap:.3f}"

logger:
  class_path: pytorch_lightning.loggers.WandbLogger
  init_args:
    project: wildfire-spread
```

### Example Data Config

```yaml
# Dataset configuration
data:
  class_path: dataloader.FireSpreadDataModule.FireSpreadDataModule
  init_args:
    data_dir: /path/to/hdf5/data
    batch_size: 15
    num_workers: 4
    pin_memory: true
    data_fold_id: 0
    temporal: false  # Single timestep
    n_leading_observations: 1
```

## 🎯 Common Use Cases

### 1. Train Official Model (Full 12-fold CV)

```bash
# Step 1: Edit wandb_12fold_cv.yaml to update data path
# Step 2: Launch sweep
wandb sweep cfgs/models/convlstm_v1/wandb_12fold_cv.yaml

# Step 3: Run agents (can parallelize across GPUs)
CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>
```

### 2. Quick Test Before Full Run

```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainers/trainer_test_short.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0
```

### 3. Run Specific Fold

```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 5 \
    --do_test true
```

### 4. Run Ablation Study (3 folds)

```bash
# Option A: Single ablation sweep
wandb sweep cfgs/models/convlstm_v1/ablations/wandb_ablation_no_attention.yaml
wandb agent <SWEEP_ID>

# Option B: All Stage 1 ablations
for ablation in baseline no_pyramid no_attention no_residual no_deep_supervision minimal; do
    wandb sweep cfgs/models/convlstm_v1/ablations/wandb_ablation_${ablation}.yaml
done
# Then run agents for each sweep
```

### 5. Compare with Original Paper Baseline

```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/ablations/original_paper_convlstm.yaml \
    --trainer cfgs/trainers/trainer_original_paper.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0 \
    --do_test true
```

## 📊 Expected Performance

### ConvLSTM v1 (Official)

| Fold | AP | UCE | Training Time |
|------|----|----|---------------|
| 0 | 0.45-0.50 | 0.15-0.20 | ~4-6h |
| ... | ... | ... | ... |
| **Mean** | **0.46 ± 0.10** | **0.18 ± 0.07** | - |

### Ablations (Stage 1)

| Config | AP | ΔAP | Insight |
|--------|----|----|---------|
| Baseline | 0.460 ± 0.101 | - | Full model |
| No Attention | 0.446 ± 0.126 | -0.014 | Most impactful |
| No Pyramid | 0.461 ± 0.152 | +0.001 | Marginal |
| Minimal | 0.452 ± 0.146 | -0.008 | Combined effect |

## 🐛 Troubleshooting

### Config File Not Found

**Error:** `FileNotFoundError: cfgs/convlstm/...`

**Solution:** Use new paths:
- Old: `cfgs/convlstm_guillermo_v1_config.yaml`
- New: `cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml`

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
--data.batch_size 8

# Or use smaller model
--config cfgs/models/convlstm_v1/ablations/stage2_small_capacity.yaml
```

### Wrong Metric Monitored

**Error:** `Early stopping conditioned on metric val_ap which is not available`

**Solution:** Use correct trainer config:
- ConvLSTM v1: `cfgs/trainers/trainer_single_gpu.yaml` (monitors val_ap)
- Original baseline: `cfgs/trainers/trainer_original_paper.yaml` (monitors val_f1)

### WandB Sweep Data Path

**Issue:** Sweep uses wrong data path

**Solution:** Edit sweep YAML file:
```yaml
parameters:
  data.data_dir:
    value: /path/to/your/hdf5/dataset  # Update this line
```

## 📚 Additional Resources

### Documentation
- [Main README](../README.md) - Repository overview
- [ConvLSTM v1 Guide](../gfx/ConvLSTM_Guillermo_v1_Guide.md) - Model architecture
- [Ablations Guide](models/convlstm_v1/ablations/README.md) - Ablation experiments
- [ConvLSTM v2 Guide](../gfx/ConvLSTM_Guillermo_v2_Guide.md) - Experimental v2

### Code Files
- Training: `src/train.py`
- Model: `src/models/ConvLSTM_Guillermo_v1.py`
- Data: `src/dataloader/FireSpreadDataModule.py`
- Metrics: `src/models/EvaluationMetrics.py`

### External Resources
- [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html)
- [WandB Sweeps](https://docs.wandb.ai/guides/sweeps)
- [WildFireSpreadTS Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)

---

**Summary:**
- ✅ Use `models/convlstm_v1/` for official experiments
- ⚠️ Avoid `models/convlstm_v2_experimental/` (not validated)
- 📖 See `models/convlstm_v1/ablations/README.md` for ablation details
- 🚀 Start with `trainer_test_short.yaml` for quick validation
- 📊 Full training: `trainer_single_gpu.yaml` + `wandb_12fold_cv.yaml`
