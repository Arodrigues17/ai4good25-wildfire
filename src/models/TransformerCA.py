import torch
import torch.nn as nn
from typing import Any, Optional
from .BaseModel import BaseModel

class TransformerCAModule(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, 
                 kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Input projection: Maps input features to d_model
        self.embedding = nn.Linear(in_channels, d_model)
        
        # Learnable positional encoding for the local window
        self.pos_embed = nn.Parameter(torch.zeros(1, kernel_size*kernel_size, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            activation='gelu', 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to binary mask logits
        # We use the center token's representation to predict the next state
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # Handle input shape: (B, T, C, H, W) -> (B, T*C, H, W) if necessary
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T*C, H, W)
        
        B, C, H, W = x.shape
        
        # Extract local patches
        # unfold: (B, C, H, W) -> (B, C*K*K, L), where L=H*W
        patches = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        
        # Reshape to (B*L, K*K, C)
        # 1. Transpose to (B, L, C*K*K)
        patches = patches.transpose(1, 2)
        # 2. View as (B, L, C, K*K) - Note: unfold flattens as channel, then spatial
        patches = patches.view(B, H*W, C, self.kernel_size*self.kernel_size)
        # 3. Permute to (B, L, K*K, C)
        patches = patches.permute(0, 1, 3, 2)
        # 4. Flatten batch and spatial: (B*H*W, K*K, C)
        # Note: We need to be careful with batch size for Transformer
        # If B*H*W is too large, it might cause issues.
        # But here the sequence length is small (K*K = 9), so it should be fine.
        # However, the batch dimension for the transformer is B*H*W, which can be huge (e.g. 16*128*128 = 262144).
        # This might be causing the CUDA error if it exceeds some internal limit.
        
        patches = patches.reshape(-1, self.kernel_size*self.kernel_size, C)
        
        # Embedding
        x = self.embedding(patches) # (B*HW, K*K, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        # Processing such a large batch size (B*H*W) in one go might be problematic.
        # Let's process in chunks if necessary, but first let's try to see if it works.
        # The error "CUDA error: invalid configuration argument" often happens when grid dimensions are too large.
        # With B*H*W ~ 260k, it might be hitting limits.
        
        # Let's try to process in chunks to avoid this.
        chunk_size = 1024 # Reduced chunk size to avoid OOM
        outputs = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            out_chunk = self.transformer(chunk)
            outputs.append(out_chunk)
        x = torch.cat(outputs, dim=0)
        
        # Take center token
        center_idx = (self.kernel_size * self.kernel_size) // 2
        x_center = x[:, center_idx, :] # (B*HW, d_model)
        
        # Output head
        logits = self.output_head(x_center) # (B*HW, 1)
        
        # Reshape back to image
        logits = logits.view(B, H, W, 1).permute(0, 3, 1, 2) # (B, 1, H, W)
        
        return logits

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
        use_doy: bool = False,
        n_leading_observations: int = 1,
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
        if use_doy:
            if flatten_temporal_dimension:
                model_in_channels = n_channels + n_leading_observations
            else:
                model_in_channels = n_channels + 1

        self.model = TransformerCAModule(
            in_channels=model_in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            kernel_size=kernel_size
        )

    def forward(self, x, doys=None):
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
            if self.hparams.flatten_temporal_dimension:
                x = x.flatten(1, 2) # (B, T*(C+1), H, W)
                
        elif self.hparams.flatten_temporal_dimension and x.ndim == 5:
             x = x.flatten(1, 2)
             
        return self.model(x)
