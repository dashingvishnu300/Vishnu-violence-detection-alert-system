"""
Suraksha-Net: Cross-Dataset Generalization Testing
==================================================
Performs generalization testing:
- Train RWF -> Test AVD
- Train RWF -> Test SFD
- Train AVD -> Test SFD
- Train SFD -> Test RWF
- Train Combined -> Test RWF / AVD / SFD

Generates the Generalization Matrix table.
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch

from datasets.video_dataset import build_dataloaders
from models.architectures import build_model
from evaluation.evaluator import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Suraksha-Net Cross-Dataset Generalization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--test-datasets", nargs="+", default=["rwf", "avd", "sfd"], help="List of datasets to evaluate against")
    parser.add_argument("--output-dir", type=str, default="results/cross_dataset", help="Output directory for reports")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    config = ckpt["config"]
    train_dataset = config["dataset"]["name"]
    model_name = config["model"]["name"]
    
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"\n========================================================")
    print(f"CROSS-DATASET GENERALIZATION MATRIX EVALUATION")
    print(f"Model: {model_name} | Trained On: {train_dataset.upper()}")
    print(f"Target Test Datasets: {[d.upper() for d in args.test_datasets]}")
    print(f"========================================================\n")
    
    matrix_rows = []
    
    for target_ds in args.test_datasets:
        print(f"\n---> Evaluating on: {target_ds.upper()} ...")
        test_cfg = dict(config)
        test_cfg["dataset"] = dict(config["dataset"])
        test_cfg["dataset"]["name"] = target_ds
        
        try:
            _, _, test_loader, info = build_dataloaders(test_cfg, dataset_name=target_ds)
        except Exception as e:
            print(f"⚠️  Skipping {target_ds}: {e}")
            continue
            
        ds_save_dir = os.path.join(args.output_dir, f"{train_dataset}_to_{target_ds}")
        results = evaluate_model(model, test_loader, device, ds_save_dir)
        m = results["metrics_at_default_0_5"]
        
        matrix_rows.append({
            "Train Dataset": train_dataset.upper(),
            "Test Dataset": target_ds.upper(),
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall (Sensitivity)": f"{m['recall']*100:.2f}%",
            "F1-Score": f"{m['f1']*100:.2f}%",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "FP Count": m["false_positives"],
            "FN Count": m["false_negatives"]
        })
        
    df = pd.DataFrame(matrix_rows)
    csv_path = os.path.join(args.output_dir, f"generalization_matrix_{train_dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n========================================================")
    print("GENERALIZATION MATRIX SUMMARY:")
    print(df.to_string(index=False))
    print(f"Saved to: {csv_path}")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()

