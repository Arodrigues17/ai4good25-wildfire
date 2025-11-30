import torch
import torch.nn as nn
from typing import Any, Optional
from .BaseModel import BaseModel

class TransformerCAModule(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Input projection: Maps input features to d_model
        # We use Conv2d with kernel_size=1 to project before unfolding
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Learnable positional encoding for the local window
        self.pos_embed = nn.Parameter(torch.zeros(1, kernel_size*kernel_size, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            activation='gelu', 
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm_final = nn.LayerNorm(d_model)

        # Output projection to binary mask logits
        # We use the center token's representation to predict the next state
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, 0)

    def forward(self, x):
        # Handle input shape: (B, T, C, H, W) -> (B, T*C, H, W) if necessary
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T*C, H, W)
        
        B, C, H, W = x.shape
        
        # Project to d_model first (B, d_model, H, W)
        x = self.embedding(x)
        
        # Extract local patches
        # unfold: (B, d_model, H, W) -> (B, d_model*K*K, L), where L=H*W
        patches = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        
        # Reshape to (B*L, K*K, d_model)
        # 1. Transpose to (B, L, d_model*K*K)
        patches = patches.transpose(1, 2)
        # 2. View as (B, L, d_model, K*K)
        patches = patches.view(B, H*W, -1, self.kernel_size*self.kernel_size)
        # 3. Permute to (B, L, K*K, d_model)
        patches = patches.permute(0, 1, 3, 2)
        # 4. Flatten batch and spatial: (B*H*W, K*K, d_model)
        x = patches.reshape(-1, self.kernel_size*self.kernel_size, x.shape[1])
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        # Processing such a large batch size (B*H*W) in one go might be problematic.
        # Let's process in chunks if necessary, but first let's try to see if it works.
        # The error "CUDA error: invalid configuration argument" often happens when grid dimensions are too large.
        # With B*H*W ~ 260k, it might be hitting limits.
        
        # Let's try to process in chunks to avoid this.
        chunk_size = 1024 # Increased chunk size since we reduced d_model
        outputs = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            out_chunk = self.transformer(chunk)
            outputs.append(out_chunk)
        x = torch.cat(outputs, dim=0)
        
        # Take center token
        center_idx = (self.kernel_size * self.kernel_size) // 2
        x_center = x[:, center_idx, :] # (B*HW, d_model)
        
        x_center = self.norm_final(x_center)

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
            kernel_size=kernel_size,
            dropout=kwargs.get('dropout', 0.1)
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
