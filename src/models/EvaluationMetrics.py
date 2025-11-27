
"""
Evaluation metrics for the wildfire spread estimation project.
The following metrics are implemented:
- Average Precision (AP)
- Expected Calibration Error (ECE)
- Unweighted Calibration Error (UCE)
And the evaluation method also plots the Precision-Recall curve and a calibration plot.

These metrics are wrapped in a custom PyTorch Lightning metric class for easy 
integration into the testing loop.

You can also directly use the `compute_metrics_and_plots` function to compute 
the metrics and plots outside of the PyTorch Lightning context.

"""
from torchmetrics import Metric
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import torch



class MyTestMetrics(Metric):
    """_summary_ Custom metric wrapper for testing purposes. 
    Computes average precision, expected calibration error (ECE),
    unweighted calibration error (UCE), and plots the precision-recall curve and calibration plot.
    """
    def __init__(self):
        super().__init__(dist_sync_on_step=True)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        self.preds.append(torch.sigmoid(preds).detach().flatten().cpu())
        self.target.append(target.detach().flatten().cpu())

    def compute(self):
        # targets = torch.cat(self.target).numpy()
        # preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.target).float().numpy()
        preds = torch.cat(self.preds).float().numpy()
        return compute_metrics_and_plots(preds, targets)
    
def compute_metrics_and_plots(preds, targets):
    """
    Computes average precision (AP), expected calibration error (ECE), unweighted calibration error (UCE),
    and plots the precision-recall curve and calibration plot."""
    precision, recall, _ = precision_recall_curve(
        targets, preds,
    )
    ap = float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))

    fig_pr = plot_pr_curve(precision, recall)

    ece, uce, fig_calibration = compute_calibration_and_plot(
        preds, targets, n_bins=15
    )

    return {
        "average_precision": ap,
        "ece": ece,
        "uce": uce,
        "fig_pr": fig_pr,
        "fig_calibration": fig_calibration
    }


def compute_calibration_and_plot(preds, target, n_bins = 15):
    """
    Produces a confidence calibration plot for binary classification, and 
    computes the Expected Calibration Error (ECE) and Unweighted Calibration Error (UCE).
    The Unweighted calibration error is the mean absolute difference between 
    the predicted probabilities and the true positive rates in each bin.
    The only difference between ECE and UCE is that ECE is weighted 
    by the number of samples in each bin which is problematic in our 
    case given the class imbalance in the dataset.
    
    (assumes the preds and target arrays are flattened)

    Args:
        y_hat (np.array): An array of predicted probabilities (from a sigmoid output).
        y (np.array): An array of true binary labels (0s and 1s).
        n_bins (int): The number of bins to use.
    """
    color_count = '#d64161'
    color_tpr = 'teal'

    y_hat_np = preds
    y_np = target
    
    # Bin the predictions based on confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_hat_np, bins) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1 # Handle max confidence edge case
    
    # Calculate metrics for each bin
    bin_acc = np.zeros(n_bins)
    bin_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        
        if np.any(bin_mask):
            bin_counts[i] = np.sum(bin_mask)
            bin_acc[i] = np.mean(y_np[bin_mask]) # Accuracy is % of true positives in the bin
            bin_prob[i] = np.mean(y_hat_np[bin_mask])


    #  Calculate ECE and UCE
    total_samples = len(y_np)
    if total_samples == 0:
      ece_val = 0
      uce_val = 0
    else:
      ece_val = np.sum(bin_counts / total_samples * np.abs(bin_acc - bin_prob))
      uce_val = np.mean(np.abs(bin_acc - bin_prob))

    #  Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Calibration Plot (Accuracy vs. Confidence)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax1.plot(bin_prob, bin_acc, marker='s', label='Model Calibration', color=color_tpr)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('True Positive Rate', color= color_tpr)
    #ax1.set_xlim(0, 1)
    #ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.tick_params(axis='y', labelcolor=color_tpr)
    
    # Create a second y-axis for the histogram
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1], bin_counts, width=1/n_bins, align='edge', edgecolor='black',color=color_count, alpha=0.3, label='Sample Count')
    ax2.set_ylabel('Count (pixels)', color=color_count)
    ax2.tick_params(axis='y', labelcolor=color_count)
    ax2.set_yscale('log')

    # Set the title with the ECE value
    fig.suptitle(f'Probability Calibration Plot | ECE: {ece_val:.4f} | UCE: {uce_val:.4f}')
    
    # Manually create a legend combining both plots
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='upper left')

    plt.tight_layout()
    
    return ece_val, uce_val, fig


def plot_pr_curve(precisions, recalls):
    """
    Plots the Precision-Recall curve.

    Args:
        precisions (np.ndarray): Array of precision values.
        recalls (np.ndarray): Array of recall values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recalls, precisions, marker='o', label='Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True)
    ax.legend()
    return fig  