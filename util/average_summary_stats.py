
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import argparse

def plot_roc_from_folder(folder):
    csv_paths = glob.glob(os.path.join(folder, "tb_logs", "*", "test_results*.csv"))
    print(csv_paths)
    if not csv_paths:
        print("No test_results.csv files found.")
        return

    all_labels = []
    all_probs = []
    all_pred_cls = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if 'labels' not in df or 'pred_probs' not in df or 'pred_class' not in df:
            print(f"Skipping {csv_path}, missing required columns.")
            continue

        # If predicted class is 0, use 1 - prob
        probs = df['pred_probs'].values
        pred_cls = df['pred_class'].values
        probs_for_class = [p if c == 1 else 1 - p for p, c in zip(probs, pred_cls)]

        all_labels.extend(df['labels'].values)
        all_probs.extend(probs_for_class)

    if not all_labels or not all_probs:
        print("No valid predictions found.")
        return

    # Compute aggregated ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Aggregated (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Aggregated ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Save
    output_path = os.path.join(folder, f'{folder.split("/")[-1]}_aggregated.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved aggregated ROC to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ROC curves from multiple test_results.csv files.')
    parser.add_argument('folder', type=str, help='Root folder containing tb_logs/*/test_results.csv')
    args = parser.parse_args()

    plot_roc_from_folder(args.folder)
