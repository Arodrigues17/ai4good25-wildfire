"""
Quick script to verify which config was used in a WandB sweep
"""

import wandb

api = wandb.Api()

# Replace with your actual sweep IDs
sweep_ids = {
    "baseline": "jfbn39m8",
    "paper_baseline": "YOUR_SWEEP_ID_HERE",  # The one you just ran
}

for name, sweep_id in sweep_ids.items():
    try:
        sweep = api.sweep(f"guillermo-ai4good/ai4good25-wildfire-src/{sweep_id}")
        print(f"\n{'='*60}")
        print(f"Sweep: {name} (ID: {sweep_id})")
        print(f"{'='*60}")

        runs = list(sweep.runs)
        if runs:
            first_run = runs[0]
            config = first_run.config

            print(f"\nModel Config:")
            print(f"  num_layers: {config.get('model.init_args.num_layers')}")
            print(f"  hidden_dims: {config.get('model.init_args.hidden_dims')}")
            print(f"  pyramid_scales: {config.get('model.init_args.pyramid_scales')}")
            print(f"  use_attention: {config.get('model.init_args.use_attention')}")
            print(f"  use_residual: {config.get('model.init_args.use_residual')}")
            print(f"  use_groupnorm: {config.get('model.init_args.use_groupnorm')}")
            print(
                f"  use_multiscale_head: {config.get('model.init_args.use_multiscale_head')}"
            )
            print(
                f"  use_feature_refinement: {config.get('model.init_args.use_feature_refinement')}"
            )
            print(f"  loss_function: {config.get('model.init_args.loss_function')}")

            print(f"\nResults (n={len(runs)} folds):")
            test_aps = [
                run.summary.get("test_AP") for run in runs if run.summary.get("test_AP")
            ]
            if test_aps:
                import numpy as np

                print(f"  test_AP: {np.mean(test_aps):.4f} ± {np.std(test_aps):.4f}")
    except Exception as e:
        print(f"\nError fetching {name}: {e}")
