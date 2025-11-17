# Prithvi-EO-2.0 Variant Guide

This guide summarizes how the Prithvi-EO-2.0 transformers that ship with TerraTorch behave inside this repository and how to keep GPU memory usage under control.

## Quick comparison

| Variant key | Params (approx.) | Transformer depth | Suggested config files | Notes |
|-------------|------------------|-------------------|------------------------|-------|
| `prithvi_eo_v2_100_tl` | 100M | 16 blocks | `cfgs/prithvi/prithvi.yaml` + `cfgs/data_multitemporal_full_features_doys.yaml` | Good default when GPU memory is limited (<16 GB). Batch size 4–8 normally fits and finetuning the entire backbone is affordable. |
| `prithvi_eo_v2_300_tl` | 300M | 24 blocks | `cfgs/prithvi/prithvi_300_tl.yaml` + `cfgs/data_multitemporal_full_features_doys.yaml` | Balanced option that still benefits from moderate batch sizes (2–4). Enable gradient checkpointing and limit the number of unfrozen blocks when memory gets tight. |
| `prithvi_eo_v2_600_tl` | 600M | 32 blocks | `cfgs/prithvi/prithvi_600_tl.yaml` + `cfgs/data_multitemporal_prithvi_600.yaml` | Requires at least a 40 GB GPU. Keep the batch size at 1, accumulate gradients to recover an effective batch of ≥2, and freeze most of the backbone unless you have multi-GPU training. |

## Implementation tips

* **Backbone indices are inferred automatically.** The `PrithviEO2Lightning` module now detects the backbone depth at runtime and selects four evenly spaced hidden states whenever `backbone_indices` is not provided. This removes the need to keep separate index lists for the 100 M, 300 M, and 600 M checkpoints while still matching the defaults used in the paper release. See `src/models/PrithviEO2Model.py` for details.
* **Set `num_frames` to match the datamodule.** The Lightning CLI already links `model.init_args.num_frames` to `data.n_leading_observations`, so you only need to override it when experimenting with different temporal windows.
* **Prefer gradient checkpointing and backbone freezing for the larger models.** The 300 M and 600 M checkpoints both benefit greatly from enabling `backbone_grad_checkpointing` and from freezing most transformer blocks (`unfreeze_backbone_blocks ≤ 6`) so that activations do not need to be stored for every layer.

## Memory-friendly hyper-parameters

* **100 M TL** – `batch_size: 4`, `accumulate_grad_batches: 1`, `freeze_backbone: false`, `unfreeze_backbone_blocks: 0`. Full finetuning is typically stable on 16 GB GPUs.
* **300 M TL** – `batch_size: 2`, `accumulate_grad_batches: 2`, `freeze_backbone: true`, `unfreeze_backbone_blocks: 8–12`. Monitor VRAM and gradually unfreeze additional blocks if there is headroom.
* **600 M TL** – `batch_size: 1`, `accumulate_grad_batches: 4`, `freeze_backbone: true`, `unfreeze_backbone_blocks: 4`. Combine with BF16 mixed precision to stay below 44 GB.

## Launch examples

```bash
# 100M TL baseline
python src/train.py \
  --config=cfgs/prithvi/prithvi.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_multitemporal_full_features_doys.yaml \
  --data.data_dir=/path/to/hdf5

# 300M TL finetuning on a 24GB GPU
python src/train.py \
  --config=cfgs/prithvi/prithvi_300_tl.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --trainer.accumulate_grad_batches=2 \
  --data=cfgs/data_multitemporal_full_features_doys.yaml \
  --data.data_dir=/path/to/hdf5

# 600M TL finetuning on a single A100 40GB
python src/train.py \
  --config=cfgs/prithvi/prithvi_600_tl.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --trainer.accumulate_grad_batches=4 \
  --data=cfgs/data_multitemporal_prithvi_600.yaml \
  --data.data_dir=/path/to/hdf5
```
