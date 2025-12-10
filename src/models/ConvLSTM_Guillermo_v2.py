from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.autograd import Variable

from .BaseModel import BaseModel
from .utae_paps_models.convlstm import ConvLSTMCell


class EnhancedChannelSpatialAttention(nn.Module):
    """Enhanced attention with both channel and spatial components."""

    def __init__(self, hidden_dim: int, reduction_ratio: int = 8):
        super(EnhancedChannelSpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduction_ratio = reduction_ratio

        # Channel attention with squeeze-excitation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // reduction_ratio, hidden_dim, 1, bias=False),
        )

        # Spatial attention with larger kernel for better context
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


class TemporalAttention(nn.Module):
    """Temporal attention to weight different timesteps - operates on temporal dimension only."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Use 1x1 convolutions for efficiency
        self.query = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.scale = self.head_dim**-0.5

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim, height, width)
        Applies temporal attention by pooling spatial dimensions first.
        """
        b, t, c, h, w = x.shape

        # Adjust head_dim if needed based on actual channel dimension
        actual_head_dim = c // self.num_heads
        if c % self.num_heads != 0:
            # If not divisible, use single head
            num_heads = 1
            actual_head_dim = c
        else:
            num_heads = self.num_heads

        # Global average pooling over spatial dimensions to get temporal features
        # (b, t, c, h, w) -> (b, t, c)
        x_temporal = F.adaptive_avg_pool2d(x.view(b * t, c, h, w), 1).view(b, t, c)

        # Reshape for multi-head attention: (b, t, c) -> (b, t, num_heads, head_dim)
        q = x_temporal.view(b, t, num_heads, actual_head_dim)
        k = x_temporal.view(b, t, num_heads, actual_head_dim)
        v = x_temporal.view(b, t, num_heads, actual_head_dim)

        # Transpose for attention: (b, num_heads, t, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention scores: (b, num_heads, t, t)
        scale = actual_head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention: (b, num_heads, t, head_dim)
        out = attn @ v

        # Reshape back: (b, num_heads, t, head_dim) -> (b, t, c)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, c)

        # Broadcast temporal attention weights to spatial dimensions
        # (b, t, c) -> (b, t, c, 1, 1) -> (b, t, c, h, w)
        temporal_weights = out.unsqueeze(-1).unsqueeze(-1)

        # Apply weighted combination to original features
        out = x * temporal_weights.sigmoid()  # Gated multiplication

        return out

        return out


class BidirectionalConvLSTMCell(nn.Module):
    """Bidirectional ConvLSTM cell processing both forward and backward."""

    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        bias=True,
        use_attention=True,
        use_residual=True,
        dropout_rate=0.0,
    ):
        super(BidirectionalConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_residual = use_residual

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Forward and backward ConvLSTM gates
        self.conv_forward = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_backward = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        # Attention mechanism
        if self.use_attention:
            self.attention = EnhancedChannelSpatialAttention(hidden_dim * 2)

        # Residual connection projection
        if self.use_residual and input_dim != hidden_dim * 2:
            self.residual_proj = nn.Conv2d(input_dim, hidden_dim * 2, 1, bias=False)
        elif self.use_residual:
            self.residual_proj = None

        # Layer normalization for stability
        self.layer_norm = nn.GroupNorm(8, hidden_dim * 2)

        # Fusion of forward and backward
        self.fusion = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1, bias=False)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def _process_direction(self, input_tensor, cur_state, conv_layer):
        """Process one direction (forward or backward)."""
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = conv_layer(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def forward(self, input_tensor, cur_state_forward, cur_state_backward):
        # Forward direction
        h_forward, c_forward = self._process_direction(
            input_tensor, cur_state_forward, self.conv_forward
        )

        # Backward direction
        h_backward, c_backward = self._process_direction(
            input_tensor, cur_state_backward, self.conv_backward
        )

        # Concatenate forward and backward
        h_next = torch.cat([h_forward, h_backward], dim=1)

        # Apply fusion
        h_next = self.fusion(h_next)

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

        # Apply dropout
        if self.dropout is not None:
            h_next = self.dropout(h_next)

        return h_next, (c_forward, c_backward)

    def init_hidden(self, batch_size, device):
        forward_state = (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )
        backward_state = (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )
        return forward_state, backward_state


class MultiScaleBidirectionalConvLSTM(nn.Module):
    """Multi-scale bidirectional ConvLSTM with enhanced pyramid pooling."""

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
        pyramid_scales=[1, 2, 4, 8],
        dropout_rate=0.0,
        stochastic_depth_rate=0.0,
        use_temporal_attention=True,
    ):
        super(MultiScaleBidirectionalConvLSTM, self).__init__()

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
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_temporal_attention = use_temporal_attention

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
            # Account for bidirectional hidden dims (2x)
            cur_input_dim = (
                input_dim if i == 0 else self.hidden_dims[i - 1] * 2
            )  # *2 for bidirectional
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

        # Build bidirectional ConvLSTM layers
        cell_list = []
        dropout_rates = [
            dropout_rate * (i + 1) / num_layers for i in range(num_layers)
        ]  # Increasing dropout

        for i in range(num_layers):
            cur_input_dim = (
                input_dim if i == 0 else self.hidden_dims[i - 1] * 2
            )  # *2 for bidirectional
            cell_list.append(
                BidirectionalConvLSTMCell(
                    input_size=input_size,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i],
                    bias=bias,
                    use_attention=use_attention,
                    use_residual=use_residual,
                    dropout_rate=dropout_rates[i],
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        # Temporal attention layers
        if use_temporal_attention:
            self.temporal_attention_layers = nn.ModuleList(
                [
                    TemporalAttention(self.hidden_dims[i] * 2, num_heads=4)
                    for i in range(num_layers)
                ]
            )

        # Skip connections from each layer to output
        self.skip_projections = nn.ModuleList(
            [
                nn.Conv2d(self.hidden_dims[i] * 2, self.hidden_dims[-1] * 2, 1)
                for i in range(num_layers - 1)
            ]
        )

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, device=input_tensor.device)

        layer_output_list = []
        last_state_list = []
        skip_connections = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_forward, h_backward = hidden_state[layer_idx]
            output_inner = []

            # Process each timestep with bidirectional cell
            h_f, c_f = h_forward
            h_b, c_b = h_backward

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

                # Stochastic depth (drop layer randomly during training)
                if (
                    self.training
                    and self.stochastic_depth_rate > 0
                    and torch.rand(1).item() < self.stochastic_depth_rate
                ):
                    # Skip this layer, just pass through (need to match output dims)
                    # Project to correct dimension if needed
                    if fused_input.shape[1] != self.cell_list[layer_idx].hidden_dim * 2:
                        # Use a simple projection
                        h_out = fused_input
                        # Pad or project to match expected output
                        if hasattr(self, "_skip_proj_" + str(layer_idx)):
                            h_out = getattr(self, "_skip_proj_" + str(layer_idx))(h_out)
                    else:
                        h_out = fused_input
                else:
                    h_out, (c_f_next, c_b_next) = self.cell_list[layer_idx](
                        fused_input, (h_f, c_f), (h_b, c_b)
                    )
                    c_f = c_f_next
                    c_b = c_b_next

                output_inner.append(h_out)

            # Stack outputs
            layer_output = torch.stack(output_inner, dim=1)

            # Apply temporal attention
            if self.use_temporal_attention:
                layer_output = self.temporal_attention_layers[layer_idx](layer_output)

            # Store skip connection (not from last layer)
            if layer_idx < self.num_layers - 1:
                skip_connections.append(
                    self.skip_projections[layer_idx](layer_output[:, -1, :, :, :])
                )

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([(h_f, c_f), (h_b, c_b)])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, skip_connections

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states


class ConvLSTM_Guillermo_v2_Seg(nn.Module):
    """
    Enhanced ConvLSTM segmentation model with:
    - Bidirectional temporal processing
    - Temporal attention
    - 4-scale pyramid pooling [1, 2, 4, 8]
    - Skip connections
    - Stochastic depth
    - Increased dropout
    """

    def __init__(
        self,
        num_classes,
        input_size,
        input_dim,
        hidden_dims,
        kernel_sizes,
        num_layers,
        pyramid_scales=[1, 2, 4, 8],
        use_attention=True,
        use_residual=True,
        dropout_rate=0.2,
        stochastic_depth_rate=0.1,
        use_temporal_attention=True,
        pad_value=0,
    ):
        super(ConvLSTM_Guillermo_v2_Seg, self).__init__()

        self.pad_value = pad_value
        self.dropout_rate = dropout_rate

        # Multi-scale bidirectional ConvLSTM encoder
        self.convlstm_encoder = MultiScaleBidirectionalConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            return_all_layers=False,
            use_attention=use_attention,
            use_residual=use_residual,
            pyramid_scales=pyramid_scales,
            dropout_rate=dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            use_temporal_attention=use_temporal_attention,
        )

        # Feature refinement layers with skip connections
        final_hidden_dim = (
            hidden_dims[-1] * 2 if isinstance(hidden_dims, list) else hidden_dims * 2
        )  # *2 for bidirectional

        # Account for skip connections
        skip_input_dim = final_hidden_dim * num_layers

        self.skip_fusion = nn.Sequential(
            nn.Conv2d(skip_input_dim, final_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(final_hidden_dim),
            nn.ReLU(inplace=True),
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

        # Multi-scale classification head with dilated convolutions
        self.classification_layers = nn.ModuleList(
            [
                nn.Conv2d(final_hidden_dim // 2, num_classes, 1),
                nn.Conv2d(final_hidden_dim // 2, num_classes, 3, padding=1),  # standard
                nn.Conv2d(
                    final_hidden_dim // 2, num_classes, 3, padding=2, dilation=2
                ),  # dilated
                nn.Conv2d(
                    final_hidden_dim // 2, num_classes, 3, padding=4, dilation=4
                ),  # more dilated
            ]
        )

        # Final fusion layer
        self.final_fusion = nn.Conv2d(num_classes * 4, num_classes, 1)

        # Deep supervision layers (for intermediate supervision)
        self.deep_supervision = nn.Conv2d(final_hidden_dim, num_classes, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        layer_output_list, last_state_list, skip_connections = self.convlstm_encoder(x)

        # Get the last time step output from the final layer
        final_output = layer_output_list[0][
            :, -1, :, :, :
        ]  # (batch, hidden_dim*2, height, width)

        # Fuse with skip connections
        if len(skip_connections) > 0:
            skip_fused = torch.cat([final_output] + skip_connections, dim=1)
            final_output = self.skip_fusion(skip_fused)

        # Deep supervision output (auxiliary loss)
        deep_sup_output = self.deep_supervision(final_output)

        # Feature refinement
        refined_features = self.feature_refinement(final_output)

        # Multi-scale classification with dilated convolutions
        multi_scale_outputs = []
        for cls_layer in self.classification_layers:
            multi_scale_outputs.append(cls_layer(refined_features))

        # Fuse multi-scale outputs
        fused_output = torch.cat(multi_scale_outputs, dim=1)
        final_prediction = self.final_fusion(fused_output)

        return final_prediction, deep_sup_output


class ConvLSTM_Guillermo_v2(BaseModel):
    """
    ConvLSTM_Guillermo_v2: Enhanced ConvLSTM with major improvements.

    Key improvements over v1:
    - Bidirectional temporal processing (forward + backward)
    - Temporal attention to weight important timesteps
    - 4-scale pyramid pooling [1, 2, 4, 8] instead of 3-scale
    - Enhanced channel + spatial attention
    - Skip connections from all layers to output
    - Stochastic depth for regularization
    - Increased dropout (0.2 default vs 0.1)
    - Dilated convolutions in classification head
    - Better feature fusion with skip connections
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
        dropout_rate: float = 0.2,
        stochastic_depth_rate: float = 0.1,
        use_temporal_attention: bool = True,
        deep_supervision_weight: float = 0.5,
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
        self.train_ap = torchmetrics.AveragePrecision(
            task="binary", compute_on_cpu=True
        )
        self.val_ap = torchmetrics.AveragePrecision(task="binary", compute_on_cpu=True)

        # Default parameters - enhanced from v1
        if kernel_sizes is None:
            kernel_sizes = [(3, 3), (3, 3), (3, 3)]
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if pyramid_scales is None:
            pyramid_scales = [1, 2, 4, 8]  # 4 scales instead of 3

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

        self.model = ConvLSTM_Guillermo_v2_Seg(
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
            stochastic_depth_rate=stochastic_depth_rate,
            use_temporal_attention=use_temporal_attention,
            pad_value=0,
        )

    def forward(self, x, doys=None):
        """Forward pass through the model."""
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        # During training, return both outputs for deep supervision
        if self.training:
            main_output, deep_sup_output = self.model(x)
            return main_output, deep_sup_output
        else:
            # During inference, only return main output
            main_output, _ = self.model(x)
            return main_output

    def get_pred_and_gt(self, batch):
        """Get predictions and ground truth, handling large test images."""
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        import math

        # Check if we need cropping (test images are larger than required size)
        if self.hparams.required_img_size is not None:
            B, T, C, H, W = x.shape
            H_req, W_req = self.hparams.required_img_size

            if x.shape[-2:] != tuple(self.hparams.required_img_size):
                # Need to crop and aggregate
                n_H = math.ceil(H / H_req)
                n_W = math.ceil(W / W_req)

                agg_output = torch.zeros(B, H, W, device=self.device)

                for i in range(n_H):
                    for j in range(n_W):
                        # Calculate crop coordinates
                        if i == n_H - 1:
                            h_start = H - H_req
                            h_end = H
                        else:
                            h_start = i * H_req
                            h_end = h_start + H_req

                        if j == n_W - 1:
                            w_start = W - W_req
                            w_end = W
                        else:
                            w_start = j * W_req
                            w_end = w_start + W_req

                        # Crop and process
                        x_crop = x[:, :, :, h_start:h_end, w_start:w_end]

                        # Get prediction for crop
                        if self.training:
                            crop_output, deep_sup_crop = self(x_crop, doys)
                            # Store deep supervision output
                            if not hasattr(self, "_deep_sup_pred"):
                                self._deep_sup_pred = deep_sup_crop
                        else:
                            crop_output = self(x_crop, doys)

                        # Accumulate in aggregated output
                        agg_output[:, h_start:h_end, w_start:w_end] = (
                            crop_output.squeeze(1)
                        )

                pred = agg_output.unsqueeze(1)
                gt = y
            else:
                # No cropping needed
                output = self(x, doys)
                if self.training and isinstance(output, tuple):
                    pred, deep_sup_pred = output
                    self._deep_sup_pred = deep_sup_pred
                else:
                    pred = output
                gt = y
        else:
            # No size requirement
            output = self(x, doys)
            if self.training and isinstance(output, tuple):
                pred, deep_sup_pred = output
                self._deep_sup_pred = deep_sup_pred
            else:
                pred = output
            gt = y

        return pred.squeeze(1), gt.squeeze(1)

    def training_step(self, batch, batch_idx):
        """Custom training step with deep supervision and AP tracking."""
        pred, gt = self.get_pred_and_gt(batch)

        # Ensure correct data types for loss computation
        pred = pred.float()
        gt = gt.float()

        # Main loss
        main_loss = self.compute_loss(pred, gt)

        # Check if we have deep supervision prediction
        if hasattr(self, "_deep_sup_pred") and self._deep_sup_pred is not None:
            # Ensure deep supervision prediction is float and squeeze if needed
            deep_sup_pred = self._deep_sup_pred.float().squeeze(1)

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
        pred_probs = torch.sigmoid(pred)
        pred_binary = (pred_probs > 0.5).long()
        gt_binary = gt.long()

        # Update metrics
        self.train_f1(pred_binary, gt_binary)
        self.train_ap(pred_probs, gt_binary)

        # Log metrics
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
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
