from typing import Any, Literal, Optional
import math
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .BaseModel import BaseModel
from terratorch.registry import BACKBONE_REGISTRY


class TemporalProjector(nn.Module):
    def __init__(
        self,
        out_channels: int,
        mode: Literal["last", "mean", "max", "conv", "attn"] = "conv",
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.attention_heads = attention_heads
        if mode == "conv":
            self.projector = nn.Sequential(
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                    groups=out_channels,
                ),
                nn.GELU(),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1),
                    groups=out_channels,
                ),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=1),
            )
        elif mode == "attn":
            if out_channels % attention_heads != 0:
                raise ValueError(
                    "Temporal attention requires the number of heads to divide the channel dimension."
                )
            self.attn = nn.MultiheadAttention(out_channels, attention_heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(out_channels)
            self.projector = None
        else:
            self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "max":
            return x.max(dim=1).values
        if self.mode == "conv":
            if self.projector is None:
                raise RuntimeError("Temporal conv projector not initialized.")
            b, t, c, h, w = x.shape
            mixed = self.projector(x.permute(0, 2, 1, 3, 4).contiguous())
            return mixed.mean(dim=2)
        if self.mode == "attn":
            b, t, c, h, w = x.shape
            tokens = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            attn_out = self.attn_norm(attn_out)
            attn_out = attn_out.mean(dim=1)
            return attn_out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported temporal pooling mode: {self.mode}")


class ForecastAwareTemporalEncoder(nn.Module):
    """Wraps the backbone's temporal encoder to support forecast-only metadata tokens."""

    def __init__(self, base_encoder: nn.Module, observed_frames: int) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.observed_frames = max(1, int(observed_frames))

    def forward(self, temporal_coords: Optional[torch.Tensor], tokens_per_frame: Optional[int] = None):
        if temporal_coords is None:
            return self.base_encoder(temporal_coords, tokens_per_frame)

        observed = temporal_coords[:, : self.observed_frames, :]
        forecast = temporal_coords[:, self.observed_frames :, :]

        if observed.shape[1] == 0:
            return self.base_encoder(temporal_coords, tokens_per_frame)

        encoding = self.base_encoder(observed, tokens_per_frame)
        if forecast.shape[1] == 0:
            return encoding

        forecast_embedding = self.base_encoder(forecast[:, :1, :], tokens_per_frame=None)
        if forecast_embedding.dim() == 2:
            forecast_embedding = forecast_embedding.unsqueeze(1)
        encoding = encoding + forecast_embedding.expand_as(encoding)
        return encoding


class _CheckpointWrapper(nn.Module):
    """Wraps a module to execute it under torch.utils.checkpoint."""

    _is_checkpoint_wrapper = True

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        def _call(*inputs):
            return self.inner(*inputs, **kwargs)

        fn = _call if kwargs else self.inner
        try:
            return checkpoint(fn, *args, use_reentrant=False)
        except TypeError:
            return checkpoint(fn, *args)


class PrithviEO2Lightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        freeze_backbone: bool = False,
        temporal_pooling: Literal["last", "mean", "max", "conv", "attn"] = "conv",
        temporal_attention_heads: int = 4,
        head_hidden_dim: int = 256,
        prithvi_variant: str = "prithvi_eo_v2_300",
        num_frames: int = 5,
        backbone_indices: Optional[list[int]] = None,
        backbone_grad_checkpointing: bool = False,
        feature_fusion: Literal["concat", "sum", "mean", "last"] = "concat",
        backbone_lr_scale: float = 1.0,
        unfreeze_backbone_blocks: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=True,
            *args,
            **kwargs,
        )
        self.freeze_backbone = freeze_backbone
        self.temporal_pooling = temporal_pooling
        self.prithvi_variant = prithvi_variant
        self.num_frames = num_frames
        self.feature_fusion = feature_fusion
        self.backbone_lr_scale = backbone_lr_scale
        self.unfreeze_backbone_blocks = max(0, int(unfreeze_backbone_blocks or 0))
        self._metadata_fallback_year = 2020.0
        self._metadata_days_per_year = 366.0
        try:
            self.backbone = BACKBONE_REGISTRY.build(
                prithvi_variant,
                pretrained=True,
                num_frames=num_frames,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Prithvi model '{prithvi_variant}' via TerraTorch. "
                "Ensure terratorch is installed (`pip install terratorch`)."
            ) from exc
        self._wrap_backbone_pos_encoding()
        self._wrap_backbone_temporal_encoder()
        self.backbone_grad_checkpointing = backbone_grad_checkpointing
        if self.backbone_grad_checkpointing:
            self._enable_backbone_grad_checkpointing()
        self._apply_backbone_freeze()
        self.backbone_depth = self._infer_backbone_depth()
        auto_indices = (
            backbone_indices
            if backbone_indices is not None
            else self._default_backbone_indices(self.backbone_depth)
        )
        self.backbone_indices = auto_indices
        if backbone_indices is None:
            depth_msg = (
                f" (depth={self.backbone_depth})" if self.backbone_depth is not None else ""
            )
            self.print(
                f"[PrithviEO2Lightning] Using automatically inferred backbone indices{depth_msg}: {self.backbone_indices}"
            )
        self.backbone_dim = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "hidden_size", 768)))
        self.patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = int(self.patch_size[0])
        patch_embed = getattr(self.backbone, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None) if patch_embed is not None else None
        self.backbone_in_channels = int(
            getattr(self.backbone, "in_channels", getattr(proj, "in_channels", n_channels))
        )
        self.input_adapter = nn.Conv3d(
            in_channels=n_channels,
            out_channels=self.backbone_in_channels,
            kernel_size=(1, 1, 1),
            bias=False,
        )
        self.temporal_projector = TemporalProjector(
            self.backbone_dim,
            mode=temporal_pooling,
            attention_heads=temporal_attention_heads,
        )
        self.channel_adapter = nn.LazyConv2d(self.backbone_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, 1, kernel_size=1),
        )
        self._last_patch_hw = None
        self._curr_input_hw = None
    
    def _apply_backbone_freeze(self) -> None:
        if not self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_backbone_blocks > 0:
            self._unfreeze_backbone_tail(self.unfreeze_backbone_blocks)

    def _resolve_backbone_blocks(self) -> Optional[nn.ModuleList]:
        blocks = getattr(self.backbone, "blocks", None)
        if isinstance(blocks, nn.ModuleList):
            return blocks
        model = getattr(self.backbone, "model", None)
        if model is not None:
            blocks = getattr(model, "blocks", None)
            if isinstance(blocks, nn.ModuleList):
                return blocks
        return None

    def _infer_backbone_depth(self) -> Optional[int]:
        blocks = self._resolve_backbone_blocks()
        if blocks is None:
            return None
        return len(blocks)

    @staticmethod
    def _default_backbone_indices(depth: Optional[int]) -> list[int]:
        """Generate evenly spaced feature indices given the backbone depth."""

        fallback = [5, 11, 17, 23]
        if depth is None or depth <= 0:
            return fallback
        if depth <= 4:
            return list(range(depth))

        quartiles = []
        for i in range(1, 5):
            position = ((i * depth) // 4) - 1
            quartiles.append(max(0, min(depth - 1, position)))
        quartiles = sorted(set(quartiles))
        if quartiles[-1] != depth - 1:
            quartiles[-1] = depth - 1
        return quartiles

    def _unfreeze_backbone_tail(self, n_blocks: int) -> None:
        blocks = self._resolve_backbone_blocks()
        if blocks is None:
            warnings.warn(
                "[PrithviEO2Lightning] Requested to unfreeze backbone blocks, "
                "but no block container was found.",
                RuntimeWarning,
            )
            return

        n_blocks = min(len(blocks), max(0, int(n_blocks)))
        if n_blocks == 0:
            return

        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        for attr in ("norm", "fc_norm", "head_norm"):
            module = getattr(self.backbone, attr, None)
            if module is None:
                module = getattr(getattr(self.backbone, "model", None), attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def _wrap_backbone_pos_encoding(self) -> None:
        def _wrap(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            interp = getattr(module, "interpolate_pos_encoding", None)
            if interp is None or getattr(interp, "_device_aligned", False):
                return
            orig_interp = interp

            def _wrapped(self, sample_shape, *args, _orig=orig_interp, **kwargs):
                pos = _orig(sample_shape, *args, **kwargs)
                if not isinstance(pos, torch.Tensor):
                    return pos
                target_device = None
                ref = getattr(self, "pos_embed", None)
                if isinstance(ref, torch.Tensor):
                    target_device = ref.device
                if target_device is None:
                    param = next(self.parameters(recurse=False), None)
                    if param is not None:
                        target_device = param.device
                if target_device is None:
                    for buffer in self.buffers(recurse=False):
                        target_device = buffer.device
                        break
                if target_device is not None and pos.device != target_device:
                    pos = pos.to(target_device, non_blocking=True)
                return pos

            _wrapped._device_aligned = True  # type: ignore[attr-defined]
            module.interpolate_pos_encoding = types.MethodType(_wrapped, module)

        _wrap(self.backbone)
        decoder = getattr(self.backbone, "decoder", None)
        _wrap(decoder)

    def _wrap_backbone_temporal_encoder(self) -> None:
        def _apply(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            encoder = getattr(module, "temporal_embed_enc", None)
            if isinstance(encoder, ForecastAwareTemporalEncoder):
                return
            if isinstance(encoder, nn.Module):
                module.temporal_embed_enc = ForecastAwareTemporalEncoder(encoder, self.num_frames)

        _apply(self.backbone)
        _apply(getattr(self.backbone, "model", None))

    def _enable_backbone_grad_checkpointing(self) -> None:
        candidates = [self.backbone, getattr(self.backbone, "model", None)]
        for module in candidates:
            if module is None:
                continue
            setter = getattr(module, "set_grad_checkpointing", None)
            if callable(setter):
                setter(True)
                return

        applied = False
        for module in candidates:
            blocks = getattr(module, "blocks", None)
            if isinstance(blocks, nn.ModuleList):
                wrapped = []
                for blk in blocks:
                    if getattr(blk, "_is_checkpoint_wrapper", False):
                        wrapped.append(blk)
                    else:
                        wrapped.append(_CheckpointWrapper(blk))
                module.blocks = nn.ModuleList(wrapped)
                applied = True

        if applied:
            return

        if not hasattr(self, "_warned_grad_checkpoint"):
            self._warned_grad_checkpoint = True
            warnings.warn(
                "[PrithviEO2Lightning] Gradient checkpointing requested but no compatible "
                "mechanism found; continuing without it.",
                RuntimeWarning,
            )

    def _backbone_patch_hw(self) -> Optional[tuple[int, int]]:
        bb = self.backbone
        candidates = (
            getattr(getattr(bb, "patch_embed", None), "patch_size", None),
            getattr(getattr(getattr(bb, "model", None), "patch_embed", None), "patch_size", None),
        )
        patch_size = next((ps for ps in candidates if ps is not None), None)
        if patch_size is None:
            proj = getattr(getattr(getattr(bb, "patch_embed", None), "proj", None), "kernel_size", None)
            if proj is None:
                proj = getattr(
                    getattr(getattr(getattr(bb, "model", None), "patch_embed", None), "proj", None),
                    "kernel_size",
                    None,
                )
            if proj is None:
                return None
            patch_size = proj
        if isinstance(patch_size, int):
            return int(patch_size), int(patch_size)
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) == 1:
                return int(patch_size[0]), int(patch_size[0])
            return int(patch_size[-2]), int(patch_size[-1])
        return None

    def _pad_to_patch_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, t, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = x.view(b * t, c, h, w)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            x = x.view(b, t, c, h + pad_h, w + pad_w)
        return x, pad_h, pad_w

    def _forward_backbone(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        if temporal_coords is not None:
            kwargs["temporal_coords"] = temporal_coords
        if location_coords is not None:
            kwargs["location_coords"] = location_coords
        runners = (
            lambda: self.backbone(pixel_values=x, **kwargs),
            lambda: self.backbone(x, **kwargs),
        )
        last_error: Optional[TypeError] = None
        for runner in runners:
            try:
                return runner()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return self.backbone(x)

    @staticmethod
    def _extract_feature(backbone_output):
        attr_candidates = (
            "pyramid",
            "feature_maps",
            "features",
            "hidden_states",
            "last_hidden_state",
            "out",
            "logits",
        )
        for attr in attr_candidates:
            if hasattr(backbone_output, attr):
                return getattr(backbone_output, attr)
        if isinstance(backbone_output, dict) and backbone_output:
            for key in attr_candidates:
                if key in backbone_output:
                    return backbone_output[key]
            return next(iter(backbone_output.values()))
        return backbone_output

    def _prepare_temporal_coords(
        self,
        temporal_coords: Optional[torch.Tensor],
        doys: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        coords = temporal_coords
        if coords is not None and not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords, dtype=dtype)
        if coords is None and doys is not None:
            if doys.dim() == 1:
                doys = doys.unsqueeze(0)
            doys = doys.to(device=device, dtype=dtype)
            base_year = torch.full_like(doys, fill_value=self._metadata_fallback_year, dtype=dtype)
            coords = torch.stack((base_year, doys - 1.0), dim=-1)
        if coords is None:
            return None
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.shape[0] != batch_size:
            coords = coords.expand(batch_size, -1, -1)
        coords = coords.to(device=device, dtype=dtype, non_blocking=True)
        coords = coords.contiguous()
        coords = self._append_forecast_temporal_token(coords)
        return coords

    def _append_forecast_temporal_token(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[1] == 0:
            return coords
        forecast = coords[:, -1:, :].clone()
        forecast[..., 1] = forecast[..., 1] + 1.0
        rollover = forecast[..., 1] >= self._metadata_days_per_year
        if rollover.any():
            forecast[..., 1] = torch.where(
                rollover,
                forecast[..., 1] - self._metadata_days_per_year,
                forecast[..., 1],
            )
            forecast[..., 0] = torch.where(
                rollover,
                forecast[..., 0] + 1.0,
                forecast[..., 0],
            )
        return torch.cat((coords, forecast), dim=1)

    def _prepare_location_coords(
        self,
        location_coords: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if location_coords is None:
            return None
        coords = location_coords
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords, dtype=dtype, device=device)
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        if coords.shape[0] != batch_size:
            coords = coords.expand(batch_size, -1)
        return coords.to(device=device, dtype=dtype, non_blocking=True)

    def _prepare_feature_map(self, feature: torch.Tensor) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor):
            raise RuntimeError("Prithvi backbone did not return tensor features.")
        if feature.dim() == 5:
            if feature.shape[1] == self.num_frames:
                pass
            elif feature.shape[2] == self.num_frames:
                feature = feature.permute(0, 2, 1, 3, 4)
            else:
                raise RuntimeError(f"Unable to identify temporal dimension for features of shape {tuple(feature.shape)}.")
            feature = self.temporal_projector(feature.contiguous())
        if feature.dim() == 3:
            feature = self._tokens_to_map(feature)
        if feature.dim() != 4:
            raise RuntimeError(f"Unexpected feature shape after processing: {feature.shape}")
        return feature

    def _gather_feature_maps(self, feature: Any) -> list[torch.Tensor]:
        if isinstance(feature, dict):
            feature = self._extract_feature(feature)
        selected = []
        if isinstance(feature, (list, tuple)):
            valid_indices = [idx for idx in self.backbone_indices if idx < len(feature)]
            if not valid_indices:
                selected = [feature[-1]]
            else:
                selected = [feature[idx] for idx in valid_indices]
        else:
            selected = [feature]
        return [self._prepare_feature_map(feat) for feat in selected]

    def _select_spatial_feature(self, feature: Any) -> torch.Tensor:
        maps = self._gather_feature_maps(feature)
        if not maps:
            raise RuntimeError("No feature maps extracted from Prithvi backbone.")
        if self.feature_fusion == "last" or len(maps) == 1:
            return maps[-1]
        target_hw = maps[-1].shape[-2:]
        resized = [
            fmap if fmap.shape[-2:] == target_hw
            else F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
            for fmap in maps
        ]
        if self.feature_fusion == "concat":
            return torch.cat(resized, dim=1)
        if self.feature_fusion == "sum":
            return torch.stack(resized, dim=0).sum(dim=0)
        if self.feature_fusion == "mean":
            return torch.stack(resized, dim=0).mean(dim=0)
        raise ValueError(f"Unsupported feature fusion mode: {self.feature_fusion}")

    def _tokens_to_map(self, tokens: torch.Tensor, remove_cls_token: bool = True) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B, L, C) tokens, got {tuple(tokens.shape)}.")
        b, seq_len, c = tokens.shape
        working = tokens
        patch_hw = self._backbone_patch_hw()
        curr_hw = getattr(self, "_curr_input_hw", None)
        gh = gw = None

        if patch_hw is not None and curr_hw is not None:
            ph, pw = patch_hw
            H, W = curr_hw
            if H % ph != 0 or W % pw != 0:
                fallback_hw = getattr(self, "_last_patch_hw", None)
                if fallback_hw is not None:
                    gh, gw = fallback_hw
                else:
                    gh = math.ceil(H / ph)
                    gw = math.ceil(W / pw)
            else:
                gh, gw = H // ph, W // pw
            patch_area = gh * gw
            if remove_cls_token and seq_len == patch_area + 1:
                working = working[:, 1:, :]
                seq_len = patch_area
            elif remove_cls_token and seq_len != patch_area and seq_len - 1 == patch_area:
                working = working[:, 1:, :]
                seq_len = patch_area
            elif seq_len != patch_area:
                gh = gw = None  # fall back to inference below

        def _infer_hw(n: int) -> Optional[tuple[int, int]]:
            for h in range(int(n ** 0.5), 0, -1):
                if n % h == 0:
                    return h, n // h
            return None

        if gh is None or gw is None:
            try_candidates = []
            if remove_cls_token and seq_len > 1:
                try_candidates.append((seq_len - 1, True))
            try_candidates.append((seq_len, False))
            for n, drop in try_candidates:
                if n <= 0:
                    continue
                hw = _infer_hw(n)
                if hw is None:
                    continue
                if drop and remove_cls_token:
                    working = working[:, 1:, :]
                    seq_len = n
                gh, gw = hw
                break
            if gh is None or gw is None:
                raise ValueError(f"Cannot infer (H, W) from token length {seq_len} after CLS handling.")

        feat = working.transpose(1, 2).reshape(b, c, gh, gw)
        self._last_patch_hw = (gh, gw)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        doys: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = x.float()
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x, pad_h, pad_w = self._pad_to_patch_size(x)
        padded_h, padded_w = x.shape[-2], x.shape[-1]
        self._last_patch_hw = (padded_h // self.patch_size, padded_w // self.patch_size)
        # x is [B, T, C, H, W] coming from your datamodule
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # -> [B, C, T, H, W]
        x = self.input_adapter(x)                   # Map dataset channels to backbone expectation
        self._curr_input_hw = tuple(int(d) for d in x.shape[-2:])
        batch_size = x.shape[0]
        backbone_temporal = self._prepare_temporal_coords(
            temporal_coords, doys, batch_size, x.device, x.dtype
        )
        backbone_location = self._prepare_location_coords(
            location_coords, batch_size, x.device, x.dtype
        )
        backbone_output = self._forward_backbone(
            x, temporal_coords=backbone_temporal, location_coords=backbone_location
        )
        feature = self._extract_feature(backbone_output)
        feature = self._select_spatial_feature(feature)
        if feature.shape[1] != self.backbone_dim:
            feature = self.channel_adapter(feature)
        logits = self.decoder(feature)
        logits = F.interpolate(logits, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
        if pad_h or pad_w:
            logits = logits[:, :, :orig_h, :orig_w]
        self._last_patch_hw = None
        self._curr_input_hw = None
        return logits

    def configure_optimizers(self):
        hparams = getattr(self, "hparams", {})

        def _fetch(name):
            if hasattr(hparams, name):
                return getattr(hparams, name)
            if isinstance(hparams, dict):
                return hparams.get(name)
            return None

        lr = _fetch("lr")
        if lr is None:
            lr = _fetch("learning_rate")
        if lr is None:
            lr = 1e-4

        weight_decay = _fetch("weight_decay")
        if weight_decay is None:
            weight_decay = 0.0

        backbone_lr_scale = _fetch("backbone_lr_scale")
        if backbone_lr_scale is None:
            backbone_lr_scale = 1.0

        head_params, backbone_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("backbone") else head_params).append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale})

        if not param_groups:
            raise RuntimeError("No trainable parameters available for optimization.")

        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
