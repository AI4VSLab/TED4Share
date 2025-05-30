
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import argparse

def plot_roc_from_folder(folder):
    # ------------------ grab paths ------------------
    csv_paths = glob.glob(os.path.join(folder, "tb_logs", "*", "test_results*.csv"))
    print(csv_paths)
    if not csv_paths:
        print("No test_results.csv files found.")
        return

    # ------------------ auroc ------------------
    all_labels = []
    all_probs = []
    all_pred_cls = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if 'labels' not in df or 'pred_probs' not in df or 'pred_class' not in df:
            print(f"Skipping {csv_path}, missing required columns.")
            continue

        # Check for NaN values
        if df[['labels', 'pred_probs', 'pred_class']].isnull().any().any():
            print(f"NaN detected in {csv_path}")

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
    print(all_labels)
    print(all_probs)
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

    # ------------------ aggerate stats ------------------
    # Find all summary CSVs
    summary_paths = glob.glob(os.path.join(folder, "tb_logs", "*", "*summary*.csv"))
    if summary_paths:
        summary_dfs = [pd.read_csv(p) for p in summary_paths]
        stacked_df = pd.concat(summary_dfs, ignore_index=True)
        avg_row = stacked_df.mean(numeric_only=True)
        std_row = stacked_df.std(numeric_only=True)
        # Convert Series to DataFrame and add a 'stat' column
        avg_row = avg_row.to_frame().T
        avg_row["stat"] = "avg"
        std_row = std_row.to_frame().T
        std_row["stat"] = "std"
        # Add 'stat' column to original data
        stacked_df["stat"] = "run"
        # Reorder columns to put 'stat' first
        cols = ["stat"] + [c for c in stacked_df.columns if c != "stat"]
        summary_out = pd.concat([stacked_df, avg_row, std_row], ignore_index=True)[cols]
        # Save
        out_csv = os.path.join(folder, f'{folder.split("/")[-1]}_aggregated_summary.csv')
        summary_out.to_csv(out_csv, index=False)
        print(f"Saved aggregated summary to: {out_csv}")
    else:
        print("No summary CSV files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ROC curves from multiple test_results.csv files.')
    parser.add_argument('folder', type=str, help='Root folder containing tb_logs/*/test_results.csv')
    args = parser.parse_args()

    plot_roc_from_folder(args.folder)
