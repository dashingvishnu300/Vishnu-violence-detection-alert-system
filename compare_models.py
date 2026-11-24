"""
Suraksha-Net: Multi-Model Benchmark & Comparison
================================================
Aggregates and formats comparison tables across:
- VGG16 Baseline
- EfficientNet-B0 Lightweight
- CNN-LSTM (ResNet18 / EfficientNet backbone)
- Optional R3D-18
Compares Accuracy, Precision, Recall, F1, ROC-AUC, Parameter Count, Size, Training Time.
"""

import os
import json
import glob
import pandas as pd
from tabulate import tabulate


def aggregate_experiment_results(exp_root: str = "experiments", output_file: str = "results/model_comparison_table.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    exp_dirs = [d for d in glob.glob(os.path.join(exp_root, "*")) if os.path.isdir(d)]
    
    rows = []
    for ed in exp_dirs:
        res_file = os.path.join(ed, "test_results.json")
        cfg_file = os.path.join(ed, "config.json")
        if not (os.path.isfile(res_file) and os.path.isfile(cfg_file)):
            continue
            
        with open(res_file, "r") as f:
            res = json.load(f)
        with open(cfg_file, "r") as f:
            cfg = json.load(f)
            
        m = res.get("metrics_at_default_0_5", {})
        rec = res.get("metrics_at_recommended_threshold", m)
        rec_th = res.get("recommended_threshold", 0.5)
        
        model_name = cfg["model"]["name"].upper()
        if "VGG" in model_name:
            backbone = "vgg16"
        elif "EFFICIENT" in model_name:
            backbone = "efficientnet_b0"
        elif "R3D" in model_name:
            backbone = "r3d_18"
        else:
            backbone = cfg["model"].get("backbone", "resnet18")
            
        rows.append({
            "Experiment": os.path.basename(ed),
            "Model": model_name,
            "Backbone": backbone,
            "Dataset": cfg["dataset"]["name"].upper(),
            "Seq Len": cfg["dataset"]["sequence_length"],
            "Accuracy": f"{m.get('accuracy', 0.0)*100:.2f}%",
            "Precision": f"{m.get('precision', 0.0)*100:.2f}%",
            "Recall (Sensitivity)": f"{m.get('recall', 0.0)*100:.2f}%",
            "F1-Score": f"{m.get('f1', 0.0)*100:.2f}%",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
            "FPR": f"{m.get('false_positive_rate', 0.0)*100:.2f}%",
            "FNR": f"{m.get('false_negative_rate', 0.0)*100:.2f}%",
            "Rec. Thresh": f"{rec_th:.2f}",
            "Rec. F1": f"{rec.get('f1', 0.0)*100:.2f}%"
        })
        
    if len(rows) == 0:
        print("No completed experiment runs found yet in experiments/.")
        return
        
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("SURAKSHA-NET: MODEL COMPARISON MATRIX")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved CSV to: {output_file}\n")


if __name__ == "__main__":
    aggregate_experiment_results()

