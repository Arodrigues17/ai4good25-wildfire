from typing import Any, Literal, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel
from terratorch.registry import BACKBONE_REGISTRY


class TemporalProjector(nn.Module):
    def __init__(self, out_channels: int, mode: Literal["last", "mean", "max", "conv"] = "conv") -> None:
        super().__init__()
        self.mode = mode
        self.projector = nn.LazyConv2d(out_channels, kernel_size=1) if mode == "conv" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "max":
            return x.max(dim=1).values
        if self.mode == "conv":
            b, t, c, h, w = x.shape
            return self.projector(x.view(b, t * c, h, w))
        raise ValueError(f"Unsupported temporal pooling mode: {self.mode}")


class PrithviEO2Lightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        freeze_backbone: bool = True,
        temporal_pooling: Literal["last", "mean", "max", "conv"] = "conv",
        head_hidden_dim: int = 256,
        prithvi_variant: str = "prithvi_eo_v2_300",
        num_frames: int = 5,
        backbone_indices: Optional[list[int]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            *args,
            **kwargs,
        )
        self.freeze_backbone = freeze_backbone
        self.temporal_pooling = temporal_pooling
        self.prithvi_variant = prithvi_variant
        self.num_frames = num_frames
        self.backbone_indices = backbone_indices or [5, 11, 17, 23]
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
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone_dim = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "hidden_size", 768)))
        self.patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = int(self.patch_size[0])
        self.temporal_projector = TemporalProjector(self.backbone_dim, mode=temporal_pooling)
        self.channel_adapter = nn.LazyConv2d(self.backbone_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_dim, 1, kernel_size=1),
        )

    def _pad_to_patch_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, t, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = x.view(b * t, c, h, w)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            x = x.view(b, t, c, h + pad_h, w + pad_w)
        return x, pad_h, pad_w

    def _forward_backbone(self, x: torch.Tensor):
        runners = (
            lambda: self.backbone(pixel_values=x),
            lambda: self.backbone(x),
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

    def _select_spatial_feature(self, feature: Any) -> torch.Tensor:
        if isinstance(feature, (list, tuple)):
            valid_indices = [idx for idx in self.backbone_indices if idx < len(feature)]
            feature = feature[valid_indices[-1]] if valid_indices else feature[-1]
        if isinstance(feature, dict):
            feature = self._extract_feature(feature)
        if not isinstance(feature, torch.Tensor):
            raise RuntimeError("Prithvi backbone did not return tensor features.")
        if feature.dim() == 5:
            feature = self.temporal_projector(feature)
        if feature.dim() == 3:
            feature = self._tokens_to_map(feature)
        if feature.dim() != 4:
            raise RuntimeError(f"Unexpected feature shape after processing: {feature.shape}")
        return feature

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        spatial_tokens = tokens
        spatial_size = int(math.sqrt(n))
        if spatial_size * spatial_size != n:
            spatial_tokens = tokens[:, 1:, :]
            n = spatial_tokens.shape[1]
            spatial_size = int(math.sqrt(n))
            if spatial_size * spatial_size != n:
                raise RuntimeError(f"Cannot reshape token sequence of length {tokens.shape[1]} into a square grid.")
        spatial_tokens = spatial_tokens[:, : spatial_size * spatial_size, :]
        return spatial_tokens.transpose(1, 2).reshape(b, c, spatial_size, spatial_size)

    def forward(self, x: torch.Tensor, doys: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = x.float()
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x, pad_h, pad_w = self._pad_to_patch_size(x)
        padded_h, padded_w = x.shape[-2], x.shape[-1]
        backbone_output = self._forward_backbone(x)
        feature = self._extract_feature(backbone_output)
        feature = self._select_spatial_feature(feature)
        if feature.shape[1] != self.backbone_dim:
            feature = self.channel_adapter(feature)
        logits = self.decoder(feature)
        logits = F.interpolate(logits, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
        if pad_h or pad_w:
            logits = logits[:, :, :orig_h, :orig_w]
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
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
