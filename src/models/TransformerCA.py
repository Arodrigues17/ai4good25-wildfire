import torch
import torch.nn as nn
from typing import Any, Optional
from .BaseModel import BaseModel

class TemporalAttention(nn.Module):
    def __init__(self, in_channels, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Conv2d(in_channels, d_model, 1)
        self.key = nn.Conv2d(in_channels, d_model, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape to (B*T, C, H, W)
        x_flat = x.view(B*T, C, H, W)
        
        # Compute Q, K, V per pixel
        Q = self.query(x_flat).view(B, T, self.d_model, H, W)
        K = self.key(x_flat).view(B, T, self.d_model, H, W)
        V = self.value(x_flat).view(B, T, C, H, W)
        
        # Attention scores: (B, H, W, T, T)
        # Q: (B, T, d, H, W) -> permute -> (B, H, W, T, d)
        Q = Q.permute(0, 3, 4, 1, 2)
        K = K.permute(0, 3, 4, 1, 2)
        V = V.permute(0, 3, 4, 1, 2)
        
        # (B, H, W, T, d) @ (B, H, W, d, T) -> (B, H, W, T, T)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # (B, H, W, T, T) @ (B, H, W, T, C) -> (B, H, W, T, C)
        out = torch.matmul(attn, V)
        
        # Permute back to (B, T, C, H, W)
        out = out.permute(0, 3, 4, 1, 2)
        
        return out

class TransformerCAModule(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, 
                 kernel_size: int = 3, dilations: list[int] = [1], dropout: float = 0.1,
                 use_temporal_attention: bool = False, n_time_steps: int = 1, input_channels_per_step: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.use_temporal_attention = use_temporal_attention
        
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(input_channels_per_step, d_model=d_model//2)

        # Padding is calculated per dilation in forward
        
        # Input projection: Maps input features to d_model
        # We use Conv2d with kernel_size=1 to project before unfolding
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Learnable positional encoding for the local window
        self.num_tokens = len(dilations) * kernel_size * kernel_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        
        # Transformer Encoder
        # Use ModuleList for Deep Supervision
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=d_model*2, 
                activation='gelu', 
                batch_first=True,
                norm_first=True,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)

        # Output heads for each layer (Deep Supervision)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.embedding.weight)
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)

    def forward(self, x, input_mask=None):
        # Handle input shape: (B, T, C, H, W) -> (B, T*C, H, W) if necessary
        if self.use_temporal_attention and x.ndim == 5:
            # Apply Temporal Attention
            x = self.temporal_attn(x)
            
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T*C, H, W)
        
        B, C, H, W = x.shape
        
        # Project to d_model first (B, d_model, H, W)
        x = self.embedding(x)
        
        all_patches = []
        for dilation in self.dilations:
            padding = (self.kernel_size - 1) * dilation // 2
            # Extract local patches
            # unfold: (B, d_model, H, W) -> (B, d_model*K*K, L), where L=H*W
            patches = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=padding, dilation=dilation)
            
            # Reshape to (B*L, K*K, d_model)
            # 1. Transpose to (B, L, d_model*K*K)
            patches = patches.transpose(1, 2)
            # 2. View as (B, L, d_model, K*K)
            patches = patches.view(B, H*W, -1, self.kernel_size*self.kernel_size)
            # 3. Permute to (B, L, K*K, d_model)
            patches = patches.permute(0, 1, 3, 2)
            # 4. Flatten batch and spatial: (B*H*W, K*K, d_model)
            patches = patches.reshape(-1, self.kernel_size*self.kernel_size, x.shape[1])
            all_patches.append(patches)
            
        x = torch.cat(all_patches, dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        # Processing such a large batch size (B*H*W) in one go might be problematic.
        # Let's process in chunks if necessary.
        
        chunk_size = 1024 
        all_layer_logits = [[] for _ in range(len(self.layers))]

        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            
            for layer_idx, layer in enumerate(self.layers):
                chunk = layer(chunk)
                
                # Compute logits for this layer
                # Take center token
                center_idx = (self.kernel_size * self.kernel_size) // 2
                chunk_center = chunk[:, center_idx, :] 
                
                # Apply norm before head
                chunk_center = self.norm_final(chunk_center)

                # Output head
                logits = self.output_heads[layer_idx](chunk_center) # (Chunk, 1)
                all_layer_logits[layer_idx].append(logits)
        
        final_outputs = []
        for layer_logits in all_layer_logits:
            l = torch.cat(layer_logits, dim=0)
            l = l.view(B, H, W, 1).permute(0, 3, 1, 2) # (B, 1, H, W)
            
            # Residual Connection
            if input_mask is not None:
                l = l + input_mask.unsqueeze(1)
                
            final_outputs.append(l)
        
        return final_outputs

class TransformerCA(BaseModel):
    def __init__(
        self, 
        flatten_temporal_dimension: bool,
        n_channels: Optional[int] = None, 
        pos_class_weight: Optional[float] = None, 
        d_model: int = 128, 
        nhead: int = 4, 
        num_layers: int = 4, 
        kernel_size: int = 3,
        dilations: list[int] = [1],
        dropout: float = 0.1,
        use_doy: bool = False,
        n_leading_observations: int = 1,
        use_temporal_attention: bool = False,
        **kwargs
    ):
        """
        Transformer Cellular Automata (TransformerCA) for Wildfire Spread Prediction.
        Uses a local Transformer to process the neighborhood of each cell.
        """
        # n_channels and pos_class_weight are injected by train.py. 
        # We provide defaults of None to satisfy the CLI parser validation, 
        # but they should be populated at runtime.
        if n_channels is None:
             raise ValueError("n_channels must be provided (should be injected by train.py)")
        if pos_class_weight is None:
             raise ValueError("pos_class_weight must be provided (should be injected by train.py)")

        # Remove pos_embed_size from kwargs if present, as it's no longer used
        kwargs.pop('pos_embed_size', None)

        super().__init__(
            n_channels=n_channels, 
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight, 
            use_doy=use_doy,
            **kwargs
        )
        self.save_hyperparameters()
        
        # Calculate actual input channels if DOY is used
        model_in_channels = n_channels
        input_channels_per_step = n_channels
        
        if use_doy:
            if flatten_temporal_dimension:
                model_in_channels = n_channels + n_leading_observations
                # If flattened, n_channels is T*C. 
                # We assume n_channels passed here is T*C if flatten_temporal_dimension is True.
                # So C = n_channels / T
                input_channels_per_step = (n_channels // n_leading_observations) + 1
            else:
                model_in_channels = n_channels + 1
                input_channels_per_step = n_channels + 1
        else:
            if flatten_temporal_dimension:
                input_channels_per_step = n_channels // n_leading_observations
            else:
                input_channels_per_step = n_channels

        self.model = TransformerCAModule(
            in_channels=model_in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            n_time_steps=n_leading_observations,
            input_channels_per_step=input_channels_per_step
        )

    def forward(self, x, doys=None):
        # Extract input mask for residual connection
        # x: (B, T, C, H, W)
        input_mask = None
        if x.ndim == 5:
            # Last channel of last timestep
            input_mask = x[:, -1, -1, :, :] # (B, H, W)

        if self.hparams.use_doy and doys is not None:
            # x: (B, T, C, H, W)
            # doys: (B, T)
            
            # Normalize DOY
            doys = doys / 366.0 # Simple normalization
            
            if x.ndim == 5:
                B, T, C, H, W = x.shape
                # Expand doys to (B, T, 1, H, W)
                doys_map = doys.view(B, T, 1, 1, 1).expand(B, T, 1, H, W)
                x = torch.cat([x, doys_map], dim=2) # (B, T, C+1, H, W)
            
            # Flatten if needed
            if self.hparams.flatten_temporal_dimension and not self.hparams.use_temporal_attention:
                x = x.flatten(1, 2) # (B, T*(C+1), H, W)
                
        elif self.hparams.flatten_temporal_dimension and x.ndim == 5:
             if not self.hparams.use_temporal_attention:
                 x = x.flatten(1, 2)
             
        outputs = self.model(x, input_mask=input_mask)
        
        if self.training:
            return outputs # List of tensors
        else:
            return outputs[-1] # Final tensor

    def training_step(self, batch, batch_idx):
        y_hat, y, x = self.get_pred_and_gt(batch)
        
        if isinstance(y_hat, list):
            total_loss = 0
            for pred in y_hat:
                total_loss += self.compute_loss(pred, y, x)
            
            # Log final layer metrics
            final_pred = y_hat[-1]
            final_loss = self.compute_loss(final_pred, y, x)
            
            self.log("train_loss", final_loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train_loss_total", total_loss.detach().item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
            f1 = self.train_f1(final_pred.detach(), y.detach())
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            return total_loss
        else:
            return super().training_step(batch, batch_idx)
    
    def get_pred_and_gt(self, batch):
        if len(batch) == 3:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None
            
        y_hat = self(x, doys)
        
        if isinstance(y_hat, list):
            y_hat = [pred.squeeze(1) for pred in y_hat]
        else:
            y_hat = y_hat.squeeze(1)
            
        return y_hat, y, x
