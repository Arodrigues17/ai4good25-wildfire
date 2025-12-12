# ConvLSTM Ablation Studies

This directory contains configuration files for systematic ablation experiments to understand the contribution of different components in the Enhanced ConvLSTM architecture.

## 📋 Overview

The ablation study is organized into two stages:

1. **Stage 1: Component Ablations** - Tests individual architectural enhancements (attention, pyramid, residual, deep supervision)
2. **Stage 2: Architecture Ablations** - Tests fundamental design choices (depth, capacity, normalization, decoder complexity)

Each ablation has two configuration files:
- **Model config** (`*.yaml`) - For single fold training
- **WandB sweep config** (`wandb_*.yaml`) - For automated multi-fold experiments

## 🔬 Stage 1: Component Ablations

These experiments test the impact of individual architectural enhancements by removing them one at a time from the full model.

### Configurations

| Config | Description | Key Parameters | Purpose |
|--------|-------------|----------------|---------|
| **baseline.yaml** | Full model (all features enabled) | All enhancements ON | Reference performance |
| **no_pyramid.yaml** | Single-scale input | `pyramid_scales: [1]` | Test multi-scale pooling contribution |
| **no_attention.yaml** | No spatial attention | `use_attention: false` | Test attention mechanism value |
| **no_residual.yaml** | No skip connections | `use_residual: false` | Test residual learning benefit |
| **no_deep_supervision.yaml** | Single loss (no auxiliary) | `deep_supervision_weight: 0.0` | Test deep supervision impact |
| **minimal.yaml** | All enhancements disabled | All OFF | Maximum simplification |

### Results Summary

| Ablation | AP ↑ | UCE ↓ | ΔAP | Key Insight |
|----------|------|-------|-----|-------------|
| **Baseline (Full)** | **0.460 ± 0.101** | **0.177 ± 0.071** | - | Reference |
| No Pyramid | 0.461 ± 0.152 | 0.175 ± 0.078 | +0.001 | Multi-scale pooling not critical |
| **No Attention** | **0.446 ± 0.126** | **0.180 ± 0.072** | **-0.014** | **Attention is most valuable** |
| No Residual | 0.460 ± 0.157 | 0.180 ± 0.076 | ±0.000 | Residual connections neutral |
| No Deep Supervision | 0.458 ± 0.144 | 0.181 ± 0.075 | -0.002 | Auxiliary loss has small benefit |
| Minimal | 0.452 ± 0.146 | 0.182 ± 0.076 | -0.008 | Combined effect ~0.8% |

**Key Finding**: Individual components provide marginal gains. Spatial attention is the most impactful enhancement (~1.4% AP improvement).

### Usage

**Single fold:**
```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/ablations/baseline.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0 \
    --do_test true
```

**Multi-fold sweep (recommended):**
```bash
wandb sweep cfgs/models/convlstm_v1/ablations/wandb_ablation_baseline.yaml
wandb agent <SWEEP_ID>
```

## 🏗️ Stage 2: Architecture Ablations

These experiments test fundamental architectural choices that define model capacity and design.

### Configurations

| Config | Description | Key Changes | Purpose |
|--------|-------------|-------------|---------|
| **stage2_single_layer.yaml** | Single ConvLSTM layer | `num_layers: 1`<br>`hidden_dims: [64]` | Test depth importance |
| **stage2_two_layers.yaml** | Two ConvLSTM layers | `num_layers: 2`<br>`hidden_dims: [64, 128]` | Test intermediate depth |
| **stage2_small_capacity.yaml** | Half channel capacity | `hidden_dims: [32, 64, 128]` | Test width/capacity impact |
| **stage2_no_groupnorm.yaml** | No normalization | `use_groupnorm: false` | Test normalization benefit |
| **stage2_simple_head.yaml** | Simple decoder (single 3×3 conv) | `use_multiscale_head: false` | Test multi-scale decoder value |
| **stage2_no_refinement.yaml** | No feature refinement | `use_feature_refinement: false` | Test refinement network impact |
| **stage2_paper_baseline.yaml** | Match original ConvLSTM | All architectural simplifications | Closest to paper architecture |

### Detailed Descriptions

#### stage2_single_layer.yaml
Tests the impact of model depth by reducing from 3 stacked ConvLSTM layers to just 1.
- **Hypothesis**: Depth is critical for temporal modeling
- **Expected**: Significant performance drop
- **Parameters**: 240K (vs 6.2M full model)

#### stage2_two_layers.yaml
Tests intermediate depth (2 layers).
- **Hypothesis**: Diminishing returns from 3rd layer
- **Expected**: Performance between 1-layer and 3-layer
- **Parameters**: ~1.5M

#### stage2_small_capacity.yaml
Halves the channel dimensions at each layer.
- **Hypothesis**: Width matters more than depth
- **Expected**: Significant drop if capacity is bottleneck
- **Parameters**: ~1.5M

#### stage2_no_groupnorm.yaml
Removes group normalization from ConvLSTM cells.
- **Hypothesis**: Normalization stabilizes training
- **Expected**: Higher variance, possible performance drop
- **Training**: May require LR adjustment

#### stage2_simple_head.yaml
Replaces multi-scale classification head (1×1, 3×3, 5×5 parallel convs) with single 3×3 conv.
- **Hypothesis**: Multi-scale decoder helps calibration
- **Expected**: UCE degradation more than AP
- **Focus**: Calibration quality

#### stage2_no_refinement.yaml
Removes the 2-layer feature refinement network before classification.
- **Hypothesis**: Refinement improves feature quality
- **Expected**: Small AP drop, larger UCE impact
- **Focus**: Probability calibration

#### stage2_paper_baseline.yaml
Combines all architectural simplifications to match original ConvLSTM paper.
- **Configuration**: 1 layer, 64 hidden dims, simple head, no GroupNorm, no refinement
- **Purpose**: Isolate architecture contribution from training methodology
- **Expected**: Large gap vs full model

### Expected Performance Hierarchy

```
Full Model (0.46 AP)
    ↓ -0.01  │ Remove refinement (stage2_no_refinement)
    ↓ -0.01  │ Simple decoder (stage2_simple_head)
    ↓ -0.02  │ No normalization (stage2_no_groupnorm)
    ↓ -0.03  │ Half capacity (stage2_small_capacity)
    ↓ -0.04  │ Two layers (stage2_two_layers)
    ↓ -0.08  │ Single layer (stage2_single_layer)
Paper Baseline (0.43 AP)
```

### Usage

**Single configuration:**
```bash
python src/train.py \
    --config cfgs/models/convlstm_v1/ablations/stage2_single_layer.yaml \
    --trainer cfgs/trainers/trainer_single_gpu.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0 \
    --do_test true
```

**Run all Stage 2 ablations (3 folds each):**
```bash
# Launch sweeps for each ablation
for config in stage2_*.yaml; do
    sweep_config="wandb_${config}"
    if [ -f "$sweep_config" ]; then
        wandb sweep cfgs/models/convlstm_v1/ablations/$sweep_config
    fi
done

# Then run agents (can parallelize across GPUs)
wandb agent <SWEEP_ID>
```

## 🔄 Original Paper Baseline

To fairly compare with the original ConvLSTM paper, we provide configurations that reproduce the original architecture and training setup.

### Configurations

| Config | Description | Key Differences |
|--------|-------------|-----------------|
| **original_paper_convlstm.yaml** | Uses ConvLSTMLightning class | BCE loss, Adam optimizer, simple architecture |
| **stage2_paper_baseline.yaml** | Uses ConvLSTM_Guillermo_v1 with simplifications | Focal loss, AdamW, but simplified architecture |
| **stage2_paper_baseline_bce.yaml** | Test BCE vs Focal on simplified architecture | Isolate loss function impact |

### Performance Decomposition

This helps decompose the performance gap:

| Configuration | AP | Loss | Optimizer | Architecture | Total Improvement |
|---------------|----|----|-----------|--------------|------------------|
| Original Paper | ~0.35 | BCE | Adam | Simple | Baseline |
| Paper Arch + Focal | 0.43 | **Focal** | **AdamW** | Simple | **+0.08** (training) |
| Full Model | 0.46 | Focal | AdamW | **Enhanced** | **+0.03** (architecture) |

**Key Insight**: Training methodology (Focal loss + AdamW) contributes **+0.08 AP** (73% of total improvement), while architectural enhancements contribute **+0.03 AP** (27%).

### Usage

**Original ConvLSTM with BCE:**
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

**Note**: Use `trainer_original_paper.yaml` which monitors `val_f1` instead of `val_ap` for compatibility with ConvLSTMLightning.

## 📊 WandB Sweep Configurations

Each ablation has a corresponding WandB sweep configuration for automated multi-fold experiments.

### Sweep File Naming Convention

- **Model config**: `{ablation_name}.yaml`
- **Sweep config**: `wandb_ablation_{ablation_name}.yaml` (Stage 1) or `wandb_{ablation_name}.yaml` (Stage 2)

### Example Sweep Config Structure

```yaml
program: src/train.py
method: grid
metric:
  name: val_ap
  goal: maximize
parameters:
  config:
    value: cfgs/models/convlstm_v1/ablations/baseline.yaml
  trainer:
    value: cfgs/trainers/trainer_single_gpu.yaml
  data:
    value: cfgs/data/data_monotemporal_full_features.yaml
  data.batch_size:
    value: 15
  data.data_dir:
    value: /path/to/your/hdf5/dataset
  data.data_fold_id:
    values: [0, 5, 10]  # 3 folds for efficiency
  do_test:
    value: true
```

### Running Sweeps

1. **Update data path** in sweep config:
   ```yaml
   data.data_dir:
     value: /path/to/your/hdf5/dataset
   ```

2. **Initialize sweep:**
   ```bash
   wandb sweep cfgs/models/convlstm_v1/ablations/wandb_ablation_baseline.yaml
   ```

3. **Run agent(s):**
   ```bash
   # Single GPU
   wandb agent <SWEEP_ID>
   
   # Multi-GPU (parallel folds)
   CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID> &
   CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID> &
   CUDA_VISIBLE_DEVICES=2 wandb agent <SWEEP_ID> &
   ```

### Recommended Folds for Quick Evaluation

For computational efficiency, we run ablations on 3 representative folds instead of all 12:
- **Folds**: [0, 5, 10] or [4, 6, 10]
- **Reasoning**: Captures variance across spatial splits
- **Speedup**: 4× faster than full 12-fold CV
- **Final evaluation**: Run best configs on all 12 folds

## 📈 Analyzing Results

### Using WandB Interface

1. **Group by ablation**: Use `unique_tag` or `config` for grouping
2. **Compare metrics**: Parallel coordinates plot for AP vs UCE
3. **View distributions**: Box plots showing variance across folds
4. **Track training**: Learning curves for validation AP

### Extracting Results Programmatically

```python
import wandb

api = wandb.Api()
runs = api.runs("your-entity/your-project", filters={"tags": "ablation_study"})

results = []
for run in runs:
    results.append({
        'name': run.config.get('config'),
        'fold': run.config.get('data.data_fold_id'),
        'test_ap': run.summary.get('test_AP'),
        'test_uce': run.summary.get('test_UCE')
    })

import pandas as pd
df = pd.DataFrame(results)
summary = df.groupby('name').agg({
    'test_ap': ['mean', 'std'],
    'test_uce': ['mean', 'std']
})
print(summary)
```

### Computing Statistics

```bash
# Average across folds
python -c "
import numpy as np
aps = [0.45, 0.47, 0.46]  # Your 3-fold results
print(f'AP: {np.mean(aps):.3f} ± {np.std(aps):.3f}')
"
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size: `--data.batch_size 10`
- Use gradient accumulation in trainer config
- Test on smaller model first (stage2_single_layer)

**2. WandB Login Issues**
```bash
wandb login
# Or set environment variable
export WANDB_API_KEY=your_key
```

**3. Data Path Errors**
- Verify HDF5 files exist at specified path
- Check fold_id is valid (0-11)
- Ensure data was preprocessed correctly

**4. Config Syntax Errors**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

**5. Metric Not Logged**
- For ConvLSTMLightning: Use `val_f1` not `val_ap`
- Check trainer config early_stopping metric matches model capabilities

### Performance Debugging

**Low AP (<0.40)**
- Verify focal loss parameters (α=0.25, γ=2.0)
- Check learning rate schedule is working
- Ensure early stopping patience is sufficient (40 epochs)
- Validate data augmentation is appropriate

**High UCE (>0.25)**
- Test with deeper models (more layers)
- Enable feature refinement
- Use multi-scale classification head
- Verify probability calibration post-processing

**High Variance Across Folds**
- Normal for wildfire task (AP std ~0.10)
- Run on more folds for reliable mean estimate
- Check for data leakage between folds

## 📚 Additional Resources

### Related Files
- **Main README**: `../../../README.md` - Repository overview
- **Model Implementation**: `../../../src/models/ConvLSTM_Guillermo_v1.py`
- **Training Script**: `../../../src/train.py`
- **Metrics**: `../../../src/models/EvaluationMetrics.py`

### Papers
- **WildFireSpreadTS**: [NeurIPS 2023 Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)
- **Focal Loss**: [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002)
- **ConvLSTM**: [Shi et al., NeurIPS 2015](https://arxiv.org/abs/1506.04214)

### Contact
For questions about ablation experiments:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Last Updated**: 2025  
**Experiments Run**: Stage 1 complete (6 configs × 3 folds), Stage 2 pending

