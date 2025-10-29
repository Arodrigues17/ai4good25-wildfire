from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.autograd import Variable

from .BaseModel import BaseModel
from .utae_paps_models.convlstm import ConvLSTMCell


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important spatial regions."""

    def __init__(self, hidden_dim: int, reduction_ratio: int = 8):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduction_ratio = reduction_ratio

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // reduction_ratio, hidden_dim, 1, bias=False),
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_conv(spatial_input)
        x = x * spatial_attention

        return x


class EnhancedConvLSTMCell(nn.Module):
    """Enhanced ConvLSTM cell with attention and residual connections."""

    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        bias=True,
        use_attention=True,
        use_residual=True,
    ):
        super(EnhancedConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_residual = use_residual

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Main ConvLSTM gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        # Attention mechanism
        if self.use_attention:
            self.attention = SpatialAttention(hidden_dim)

        # Residual connection projection
        if self.use_residual and input_dim != hidden_dim:
            self.residual_proj = nn.Conv2d(input_dim, hidden_dim, 1, bias=False)
        elif self.use_residual:
            self.residual_proj = None

        # Layer normalization for stability
        self.layer_norm = nn.GroupNorm(8, hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # Apply layer normalization
        h_next = self.layer_norm(h_next)

        # Apply attention
        if self.use_attention:
            h_next = self.attention(h_next)

        # Apply residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(input_tensor)
            else:
                residual = input_tensor
            h_next = h_next + residual

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )


class MultiScaleConvLSTM(nn.Module):
    """Multi-scale ConvLSTM with pyramid pooling and feature fusion."""

    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dims,
        kernel_sizes,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
        use_attention=True,
        use_residual=True,
        pyramid_scales=[1, 2, 4],
    ):
        super(MultiScaleConvLSTM, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dims = (
            hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        )
        self.kernel_sizes = (
            kernel_sizes
            if isinstance(kernel_sizes, list)
            else [kernel_sizes] * num_layers
        )
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.pyramid_scales = pyramid_scales

        # Build multi-scale pyramid pooling layers
        self.pyramid_pools = nn.ModuleList(
            [
                nn.AdaptiveAvgPool2d((self.height // scale, self.width // scale))
                for scale in pyramid_scales
            ]
        )

        # Feature fusion after pyramid pooling - one for each layer
        self.pyramid_fusions = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i - 1]
            pyramid_channels = cur_input_dim * len(pyramid_scales)
            self.pyramid_fusions.append(
                nn.Sequential(
                    nn.Conv2d(
                        pyramid_channels, cur_input_dim, 3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(cur_input_dim),
                    nn.ReLU(inplace=True),
                )
            )

        # Build ConvLSTM layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                EnhancedConvLSTMCell(
                    input_size=input_size,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i],
                    bias=bias,
                    use_attention=use_attention,
                    use_residual=use_residual,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # Apply multi-scale pyramid pooling
                current_input = cur_layer_input[:, t, :, :, :]
                _, _, spatial_h, spatial_w = current_input.shape
                pyramid_features = []

                for pool in self.pyramid_pools:
                    pooled = pool(current_input)
                    upsampled = F.interpolate(
                        pooled,
                        size=(spatial_h, spatial_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pyramid_features.append(upsampled)

                # Fuse pyramid features
                pyramid_concat = torch.cat(pyramid_features, dim=1)
                fused_input = self.pyramid_fusions[layer_idx](pyramid_concat)

                h, c = self.cell_list[layer_idx](fused_input, [h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states


class ConvLSTM_Guillermo_v1_Seg(nn.Module):
    """Advanced ConvLSTM segmentation model with multi-scale processing and attention."""

    def __init__(
        self,
        num_classes,
        input_size,
        input_dim,
        hidden_dims,
        kernel_sizes,
        num_layers,
        pyramid_scales=[1, 2, 4],
        use_attention=True,
        use_residual=True,
        dropout_rate=0.1,
        pad_value=0,
    ):
        super(ConvLSTM_Guillermo_v1_Seg, self).__init__()

        self.pad_value = pad_value
        self.dropout_rate = dropout_rate

        # Multi-scale ConvLSTM encoder
        self.convlstm_encoder = MultiScaleConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            return_all_layers=False,
            use_attention=use_attention,
            use_residual=use_residual,
            pyramid_scales=pyramid_scales,
        )

        # Feature refinement layers
        final_hidden_dim = (
            hidden_dims[-1] if isinstance(hidden_dims, list) else hidden_dims
        )
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(final_hidden_dim, final_hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(final_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(
                final_hidden_dim, final_hidden_dim // 2, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(final_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

        # Multi-scale classification head
        self.classification_layers = nn.ModuleList(
            [
                nn.Conv2d(final_hidden_dim // 2, num_classes, 1),
                nn.Conv2d(final_hidden_dim // 2, num_classes, 3, padding=1),
                nn.Conv2d(final_hidden_dim // 2, num_classes, 5, padding=2),
            ]
        )

        # Final fusion layer
        self.final_fusion = nn.Conv2d(num_classes * 3, num_classes, 1)

        # Deep supervision layers (for intermediate supervision)
        self.deep_supervision = nn.Conv2d(final_hidden_dim, num_classes, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        layer_output_list, last_state_list = self.convlstm_encoder(x)

        # Get the last time step output from the final layer
        final_output = layer_output_list[0][
            :, -1, :, :, :
        ]  # (batch, hidden_dim, height, width)

        # Deep supervision output (auxiliary loss)
        deep_sup_output = self.deep_supervision(final_output)

        # Feature refinement
        refined_features = self.feature_refinement(final_output)

        # Multi-scale classification
        multi_scale_outputs = []
        for cls_layer in self.classification_layers:
            multi_scale_outputs.append(cls_layer(refined_features))

        # Fuse multi-scale outputs
        fused_output = torch.cat(multi_scale_outputs, dim=1)
        final_prediction = self.final_fusion(fused_output)

        return final_prediction, deep_sup_output


class ConvLSTM_Guillermo_v1(BaseModel):
    """
    ConvLSTM_Guillermo_v1: An advanced ConvLSTM model for wildfire spread prediction.

    Features:
    - Multi-scale pyramid pooling for capturing features at different scales
    - Spatial attention mechanism for focusing on important regions
    - Residual connections for better gradient flow
    - Layer normalization for training stability
    - Deep supervision for better feature learning
    - Multi-scale classification head for robust predictions
    - Enhanced ConvLSTM cells with attention and normalization
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str,
        img_height_width: Tuple[int, int],
        kernel_sizes: List[Tuple[int, int]] = None,
        hidden_dims: List[int] = None,
        num_layers: int = 3,
        pyramid_scales: List[int] = None,
        use_attention: bool = True,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
        deep_supervision_weight: float = 0.3,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            required_img_size=img_height_width,
            *args,
            **kwargs
        )
        self.save_hyperparameters()

        # Add Average Precision metrics for training and validation
        # Using compute_on_cpu=True to reduce GPU memory usage
        self.train_ap = torchmetrics.AveragePrecision(
            task="binary", compute_on_cpu=True
        )
        self.val_ap = torchmetrics.AveragePrecision(task="binary", compute_on_cpu=True)

        # Default parameters
        if kernel_sizes is None:
            kernel_sizes = [(3, 3), (3, 3), (3, 3)]
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if pyramid_scales is None:
            pyramid_scales = [1, 2, 4]

        self.deep_supervision_weight = deep_supervision_weight

        # Ensure we have the right number of parameters for each layer
        if len(kernel_sizes) != num_layers:
            kernel_sizes = (
                kernel_sizes * num_layers
                if len(kernel_sizes) == 1
                else kernel_sizes[:num_layers]
            )
        if len(hidden_dims) != num_layers:
            hidden_dims = (
                hidden_dims * num_layers
                if len(hidden_dims) == 1
                else hidden_dims[:num_layers]
            )

        self.model = ConvLSTM_Guillermo_v1_Seg(
            num_classes=1,
            input_size=img_height_width,
            input_dim=n_channels,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            pyramid_scales=pyramid_scales,
            use_attention=use_attention,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
        )

    def forward(self, x, doys=None):
        """Forward pass through the model."""
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        # The model returns both main prediction and deep supervision output
        main_output, deep_sup_output = self.model(x)

        # During training, we'll use both outputs for loss computation
        # During inference, we only return the main output
        if self.training:
            return main_output, deep_sup_output
        else:
            return main_output

    def get_pred_and_gt(self, batch):
        """Override to handle tuple output during training."""
        # UTAE and TSViT use an additional doy feature as input.
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        # Get model prediction
        output = self(x, doys)

        # Handle tuple output during training
        if isinstance(output, tuple):
            main_pred, deep_sup_pred = output
            # Store deep supervision prediction for use in training_step
            self._deep_sup_pred = (
                deep_sup_pred.squeeze(1) if deep_sup_pred.dim() > 3 else deep_sup_pred
            )
            y_hat = main_pred.squeeze(1) if main_pred.dim() > 3 else main_pred
        else:
            y_hat = output.squeeze(1) if output.dim() > 3 else output
            self._deep_sup_pred = None

        return y_hat, y

    def training_step(self, batch, batch_idx):
        """Training step with deep supervision."""
        pred, gt = self.get_pred_and_gt(batch)

        # Ensure correct data types for loss computation
        pred = pred.float()
        gt = gt.float()

        # Main loss
        main_loss = self.compute_loss(pred, gt)

        # Check if we have deep supervision prediction
        if hasattr(self, "_deep_sup_pred") and self._deep_sup_pred is not None:
            # Ensure deep supervision prediction is float
            deep_sup_pred = self._deep_sup_pred.float()

            # Deep supervision loss
            deep_sup_loss = self.compute_loss(deep_sup_pred, gt)

            # Combined loss
            total_loss = main_loss + self.deep_supervision_weight * deep_sup_loss

            # Log losses
            self.log(
                "train/main_loss", main_loss, on_step=True, on_epoch=True, prog_bar=True
            )
            self.log("train/deep_sup_loss", deep_sup_loss, on_step=True, on_epoch=True)
            self.log(
                "train/total_loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            total_loss = main_loss
            self.log(
                "train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
            )

        # Compute metrics
        pred_probs = torch.sigmoid(pred)  # Get probabilities for AP
        pred_binary = (pred_probs > 0.5).long()
        gt_binary = gt.long()

        # Update metrics - F1 for step-level, AP only for epoch-level
        self.train_f1(pred_binary, gt_binary)
        self.train_ap(pred_probs, gt_binary)

        # Log F1 at both step and epoch level
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        # Log AP only at epoch level to save memory
        self.log(
            "train/ap", self.train_ap, on_step=False, on_epoch=True, prog_bar=False
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with Average Precision tracking."""
        pred, gt = self.get_pred_and_gt(batch)

        # Ensure correct data types
        pred = pred.float()
        gt = gt.float()

        # Compute loss
        val_loss = self.compute_loss(pred, gt)

        # Compute metrics
        pred_probs = torch.sigmoid(pred)
        pred_binary = (pred_probs > 0.5).long()
        gt_binary = gt.long()

        # Update metrics
        self.val_f1(pred_binary, gt_binary)
        self.val_ap(pred_probs, gt_binary)

        # Log metrics
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss
