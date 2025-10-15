from typing import Any, Dict, Literal, Optional
from transformers import AutoConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .BaseModel import BaseModel


class TemporalProjector(nn.Module):
    def __init__(
        self,
        out_channels: int,
        mode: Literal["last", "mean", "conv"] = "conv",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "conv":
            self.projector = nn.LazyConv2d(out_channels, kernel_size=1)
        else:
            self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "conv":
            b, t, c, h, w = x.shape
            x = x.view(b, t * c, h, w)
            return self.projector(x)
        raise ValueError(f"Unsupported temporal pooling mode: {self.mode}")


class PrithviEO2Lightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        prithvi_model_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        freeze_backbone: bool = True,
        temporal_pooling: Literal["last", "mean", "conv"] = "conv",
        head_hidden_dim: int = 256,
        output_dropout: float = 0.1,
        trust_remote_code: bool = True,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.backbone_kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        self.backbone_kwargs.setdefault("num_labels", 1)
        self.backbone_checkpoint = prithvi_model_name
        self.trust_remote_code = trust_remote_code
        config_init_kwargs = {"num_labels": self.backbone_kwargs["num_labels"]}

        self.backbone_config = AutoConfig.from_pretrained(
            self.backbone_checkpoint,
            trust_remote_code=self.trust_remote_code,
            **config_init_kwargs,
        )
        if getattr(self.backbone_config, "num_labels", None) in (None, 0):
            self.backbone_config.num_labels = self.backbone_kwargs["num_labels"]
            self.backbone_config.id2label = {
                i: f"LABEL_{i}" for i in range(self.backbone_config.num_labels)
            }
            self.backbone_config.label2id = {
                v: k for k, v in self.backbone_config.id2label.items()
            }
        try:
            self.backbone = AutoModel.from_pretrained(
                self.backbone_checkpoint,
                config=self.backbone_config,
                trust_remote_code=self.trust_remote_code,
                **{k: v for k, v in self.backbone_kwargs.items() if k != "num_labels"},
            )
        except OSError as err:
            missing_weights = "does not appear to have a file named" in str(err)
            if missing_weights:
                raise RuntimeError(
                    f"Failed to download weights for '{self.backbone_checkpoint}'. "
                    "Ensure you have accepted the model license and that a valid Hugging Face token "
                    "is available (e.g., run `huggingface-cli login`)."
                ) from err
            raise
        image_size = getattr(self.backbone.config, "image_size", None)
        required_size = (
            (image_size, image_size)
            if isinstance(image_size, int)
            else None
        )
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            required_img_size=required_size,
            **kwargs,
        )
        self.expected_channels = getattr(
            self.backbone.config,
            "num_channels",
            getattr(self.backbone.config, "in_channels", n_channels),
        )
        self.patch_size = getattr(self.backbone.config, "patch_size", 16)
        self.temporal_adapter = TemporalProjector(
            out_channels=self.expected_channels,
            mode=temporal_pooling,
        )
        if temporal_pooling == "conv":
            self.channel_adapter = nn.Identity()
        else:
            self.channel_adapter = nn.LazyConv2d(
                self.expected_channels,
                kernel_size=1,
            )
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_size, head_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Conv2d(head_hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        projected = self.temporal_adapter(x)
        projected = (
            projected
            if isinstance(self.channel_adapter, nn.Identity)
            else self.channel_adapter(projected)
        )
        orig_h, orig_w = projected.shape[-2:]
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            projected = F.pad(projected, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = projected.shape[-2:]
        outputs = self.backbone(pixel_values=projected)
        tokens = outputs.last_hidden_state[:, 1:, :]
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size
        tokens = tokens[:, : grid_h * grid_w, :]
        tokens = tokens.transpose(1, 2).reshape(
            projected.shape[0],
            self.hidden_size,
            grid_h,
            grid_w,
        )
        logits = self.head(tokens)
        logits = F.interpolate(
            logits,
            size=(padded_h, padded_w),
            mode="bilinear",
            align_corners=False,
        )
        if pad_h or pad_w:
            logits = logits[:, :, :orig_h, :orig_w]
        return logits
