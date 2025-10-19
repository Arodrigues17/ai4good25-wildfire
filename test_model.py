#!/usr/bin/env python3
"""
Simple test script to verify ConvLSTM_Guillermo_v1 can be instantiated properly.
Run this to debug model instantiation issues.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from models.ConvLSTM_Guillermo_v1 import ConvLSTM_Guillermo_v1

    print("✓ ConvLSTM_Guillermo_v1 imported successfully")

    # Test model instantiation with the same parameters from your config
    model = ConvLSTM_Guillermo_v1(
        n_channels=40,
        flatten_temporal_dimension=False,
        pos_class_weight=236,
        loss_function="Focal",
        img_height_width=[128, 128],
        kernel_sizes=[[3, 3], [3, 3], [3, 3]],
        hidden_dims=[64, 128, 256],
        num_layers=3,
        pyramid_scales=[1, 2, 4],
        use_attention=True,
        use_residual=True,
        dropout_rate=0.1,
        deep_supervision_weight=0.3,
    )
    print("✓ Model instantiated successfully")
    print(f"✓ Model type: {type(model)}")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Model instantiation error: {e}")
    import traceback

    traceback.print_exc()
