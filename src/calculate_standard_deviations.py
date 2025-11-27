import wandb
import statistics
from collections import defaultdict

ENTITY = "bluefrog-university-of-zurich" 
PROJECT = "ai4good25-wildfire-src"
SWEEP_IDS = ["serlb54o", "amgs0fsy", "hqgpmiai"] 
METRICS_TO_ANALYZE = ["test_AP", "test_UCE"] 

api = wandb.Api()

print(f"--- Analyzing Metrics from Project: {ENTITY}/{PROJECT} ---")

def analyze_sweep(sweep_id):
    """Fetches runs for a sweep, extracts metric values, and calculates stats."""
    full_sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
    
    try:
        sweep = api.sweep(full_sweep_path)
        print(f"\n## Sweep ID: {sweep_id} ({sweep.name})")

        runs = sweep.runs
        
        if not runs:
            print("No runs found for this sweep.")
            return

        metric_values = defaultdict(list)

        for run in runs:
            if run.state == "finished" and run.summary:
                for metric in METRICS_TO_ANALYZE:
                    if metric in run.summary:
                        metric_values[metric].append(run.summary[metric])
        
        for metric in METRICS_TO_ANALYZE:
            values = metric_values.get(metric, [])
            num_runs = len(values)

            if num_runs > 0:
                mean_val = statistics.mean(values)
                
                std_dev_val = statistics.stdev(values) if num_runs > 1 else 0.0

                print(f"--- {metric} ---")
                print(f"  Runs Analyzed: **{num_runs}**")
                print(f"  Mean: **{mean_val:.4f}**")
                print(f"  Standard Deviation (Std Dev): **{std_dev_val:.4f}**")
            else:
                print(f"--- {metric} ---")
                print(f"  No valid final summary data found for {metric} in any run.")

    except wandb.CommError as e:
        print(f"Error accessing sweep {sweep_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for sweep {sweep_id}: {e}")

for sweep_id in SWEEP_IDS:
    analyze_sweep(sweep_id)

print("\n--- Script finished ---")
