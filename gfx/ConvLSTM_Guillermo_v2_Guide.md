# ConvLSTM_Guillermo_v2 - Experimental Development (NOT RELEASED)

> ⚠️ **EXPERIMENTAL MODEL - NOT PART OF FINAL SUBMISSION**  
> This model contains additional enhancements that were explored but not fully tested or validated.  
> For the **official released model**, see [ConvLSTM_Guillermo_v1_Guide.md](ConvLSTM_Guillermo_v1_Guide.md).  
> v2 is provided for future research and development but is not included in the paper submission.

## Overview
ConvLSTM_Guillermo_v2 is an experimental enhancement of the v1 model with additional architectural features that showed promise but require further development and validation. Based on initial v1 analysis (val_ap=0.48, train_ap=0.54), v2 explores more aggressive architectural changes.

## Key Improvements Over v1

### 1. **Bidirectional Temporal Processing** 
- **v1**: Unidirectional ConvLSTM (forward only)
- **v2**: Bidirectional ConvLSTM processing sequences forward AND backward
- **Benefit**: Captures temporal context from both directions, better understanding of fire progression patterns

### 2. **Temporal Attention Mechanism** 
- **New Feature**: Multi-head temporal attention layer
- Learns which timesteps are most important for prediction
- Fire spread dynamics vary over time - some moments are more critical than others
- 4 attention heads for diverse temporal pattern recognition

### 3. **Enhanced Multi-Scale Pyramid Pooling** 
- **v1**: 3 scales [1, 2, 4]
- **v2**: 4 scales [1, 2, 4, 8]
- **Benefit**: Captures larger fire spread patterns and contextual information

### 4. **Skip Connections & Feature Fusion** 
- All encoder layers now contribute to final prediction via skip connections
- Better gradient flow and multi-level feature integration
- Reduces information loss through the network depth

### 5. **Stochastic Depth Regularization** 
- Randomly drops ConvLSTM layers during training (rate: 0.1)
- Prevents over-reliance on specific layers
- Improves generalization and reduces overfitting

### 6. **Dilated Convolutions in Classification** 
- Classification head uses dilated convolutions [1, 2, 4]
- Increases receptive field without losing resolution
- Better captures irregular fire spread patterns

### 7. **Enhanced Loss Function** 
- **v1**: Focal loss only
- **v2**: Combined FocalDice loss (0.5 Focal + 0.5 Dice)
- **Benefit**: Balances pixel-wise accuracy (Focal) with region-based IoU (Dice)
- Better for highly imbalanced segmentation

### 8. **Improved Regularization** 
- **Dropout**: Increased from 0.1 → 0.2 (progressive: higher in deeper layers)
- **Weight Decay**: Added 0.01 to AdamW optimizer
- **Deep Supervision**: Weight increased from 0.3 → 0.5

### 9. **Advanced Learning Rate Schedule** 
- **v1**: Fixed lr=0.001
- **v2**: Cosine annealing with warm restarts
  - Initial lr: 0.003 (3x higher)
  - Restarts every 10 epochs (then 20, 40...)
  - Min lr: 0.0001
- **Benefit**: Helps escape local minima, periodic restarts explore new regions

## Architecture Details

### Model Components
```
Input (B, T=5, C=40, H=128, W=128)
    ↓
Multi-Scale Pyramid Pooling [1, 2, 4, 8]
    ↓
Bidirectional ConvLSTM Layers (3 layers: 64→128→256 channels)
    ├─ Forward LSTM
    ├─ Backward LSTM
    ├─ Enhanced Channel+Spatial Attention
    ├─ Residual Connections
    ├─ Layer Normalization
    └─ Stochastic Depth
    ↓
Temporal Attention (4 heads)
    ↓
Skip Connections Fusion
    ↓
Feature Refinement (dropout=0.2)
    ↓
Multi-Scale Classification (1x1, 3x3, dilated 3x3 @d=2, dilated 3x3 @d=4)
    ↓
Output (B, 1, H=128, W=128)

Deep Supervision Branch (auxiliary output for training)
```

### Parameter Count
- **v1**: ~6.2M parameters
- **v2**: ~12-15M parameters (bidirectional doubles hidden state sizes)

## Configuration Files

### Model Config: `cfgs/convlstm_guillermo_v2_config.yaml`
```yaml
optimizer: AdamW (lr=0.003, weight_decay=0.01)
lr_scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
loss_function: FocalDice (50/50 split)
hidden_dims: [64, 128, 256]
pyramid_scales: [1, 2, 4, 8]
dropout_rate: 0.2
stochastic_depth_rate: 0.1
deep_supervision_weight: 0.5
```

### Trainer Configs
1. **`trainer_v2_test_short.yaml`** - Quick 5-epoch test (20% data)
2. **`trainer_v2_full.yaml`** - Full 200-epoch training with early stopping (patience=25)

## Running the Model

### Quick Test (Recommended First)
```bash
python src/train.py \
  --config=cfgs/convlstm_guillermo_v2_config.yaml \
  --trainer=cfgs/trainer_v2_test_short.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --data.batch_size 12 \
  --data.load_from_hdf5 true \
  --data.data_dir ../data/hdf5
```
**Note**: Batch size reduced to 12 (from 15) due to increased model size.

### Full Training
```bash
python src/train.py \
  --config=cfgs/convlstm_guillermo_v2_config.yaml \
  --trainer=cfgs/trainer_v2_full.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --data.batch_size 12 \
  --data.load_from_hdf5 true \
  --data.data_dir ../data/hdf5
```

## Expected Improvements

Based on the implemented enhancements, expected performance gains:

1. **Average Precision (AP)**: +5-15% improvement
   - v1 baseline: val_ap=0.48
   - v2 target: val_ap=0.52-0.55

2. **Convergence Speed**: Faster due to:
   - Higher initial learning rate
   - Better gradient flow (skip connections)
   - Bidirectional processing

3. **Generalization**: Better test performance due to:
   - Stochastic depth
   - Increased dropout
   - Combined loss function

4. **Spatial Accuracy**: Improved fire boundary detection from:
   - Dilated convolutions
   - 4-scale pyramid
   - Enhanced attention

## Memory Considerations

v2 uses ~1.5-2x more memory than v1 due to:
- Bidirectional processing (2x hidden states)
- 4 pyramid scales instead of 3
- Skip connections storage

**Recommendations**:
- Use batch_size=12 instead of 15
- Monitor GPU memory during training
- Reduce to batch_size=10 if OOM errors occur

## Monitoring Training

Key metrics to watch in WandB:

1. **train/ap & val_ap**: Primary metrics (should increase steadily)
2. **train/f1 & val_f1**: Binary classification performance
3. **train/main_loss & train/deep_sup_loss**: Component losses
4. **lr**: Learning rate schedule (cosine annealing pattern)

## Next Steps After v2

If v2 still needs improvement, consider:

1. **Data Augmentation**: Spatial (rotations, flips) and temporal
2. **Ensemble**: Train 3-5 v2 models with different seeds
3. **Threshold Optimization**: Find optimal prediction threshold on validation set
4. **Test-Time Augmentation**: Average predictions from augmented inputs
5. **Feature Engineering**: Add derived features (fire progression rate, distance from edge)

## Files Created

New files in this implementation:
- `src/models/ConvLSTM_Guillermo_v2.py` - Enhanced model architecture
- `cfgs/convlstm_guillermo_v2_config.yaml` - Model configuration
- `cfgs/trainer_v2_full.yaml` - Full training configuration
- `cfgs/trainer_v2_test_short.yaml` - Quick test configuration
- Modified: `src/models/BaseModel.py` - Added FocalDice combined loss

## Troubleshooting

### Out of Memory
- Reduce batch_size to 10 or 8
- Set `stochastic_depth_rate: 0.15` (more aggressive dropping)
- Reduce hidden_dims to [64, 128, 192]

### Training Instability
- Reduce initial lr to 0.002
- Increase gradient_clip_val to 0.5
- Check for NaN values in logs

### Poor Convergence
- Increase deep_supervision_weight to 0.6
- Try focal_dice_weight: 0.4 (more Dice, less Focal)
- Increase patience to 30

## Citation

Based on improvements suggested from analyzing:
- Medical image segmentation literature (U-Net++, Attention U-Net)
- Temporal modeling advances (Temporal Attention, BiLSTM)
- Fire spread prediction challenges (class imbalance, irregular patterns)
