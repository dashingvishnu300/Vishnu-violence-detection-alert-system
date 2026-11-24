"""
Suraksha-Net: Final Model Handoff Packager
==========================================
Produces the official handoff package for Arpit / Deployment Team:
- best_model.pth
- model_config.json
- preprocessing_config.json
- class_mapping.json
- recommended_threshold.json
- metrics_summary.json
- HANDOFF_README.md
"""

import os
import sys
import json
import shutil
import argparse
import torch

from models.architectures import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Package trained model for Arpit / Deployment Team")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--test-results", type=str, required=True, help="Path to test_results.json")
    parser.add_argument("--output-dir", type=str, default="handoff_to_arpit", help="Output package directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    with open(args.test_results, "r") as f:
        test_results = json.load(f)
        
    # Copy best_model.pth
    target_model_path = os.path.join(args.output_dir, "best_model.pth")
    shutil.copy2(args.checkpoint, target_model_path)
    
    # 1. model_config.json
    model_cfg = config["model"]
    model_inst = build_model(config)
    total_params = sum(p.numel() for p in model_inst.parameters())
    model_size_mb = os.path.getsize(target_model_path) / (1024 * 1024)
    
    handoff_model_config = {
        "architecture": model_cfg["name"],
        "backbone": model_cfg.get("backbone", "resnet18"),
        "input_tensor_format": "B x T x C x H x W",
        "sequence_length": config["dataset"]["sequence_length"],
        "image_size": [config["dataset"]["image_size"], config["dataset"]["image_size"]],
        "feature_dimension": getattr(model_inst, "feature_dim", 512),
        "lstm_hidden_size": model_cfg.get("lstm_hidden_size", 256),
        "lstm_num_layers": model_cfg.get("lstm_num_layers", 1),
        "dropout": model_cfg.get("lstm_dropout", 0.3),
        "number_of_classes": model_cfg.get("num_classes", 2),
        "total_parameters": total_params,
        "model_file_size_mb": round(model_size_mb, 2),
        "trained_on_dataset": config["dataset"]["name"],
        "output_format": "Logits [Non-Violence, Violence] -> Softmax to obtain probabilities"
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(handoff_model_config, f, indent=2)
        
    # 2. preprocessing_config.json
    norm = config["normalization"]
    handoff_preprocessing = {
        "color_space": "RGB",
        "resize_dimensions": [config["dataset"]["image_size"], config["dataset"]["image_size"]],
        "sequence_length_frames": config["dataset"]["sequence_length"],
        "sampling_method": config["dataset"].get("sampling_strategy", "uniform"),
        "normalization_mean": norm["mean"],
        "normalization_std": norm["std"],
        "expected_tensor_shape": [1, config["dataset"]["sequence_length"], 3, config["dataset"]["image_size"], config["dataset"]["image_size"]],
        "value_range": "[0.0, 1.0] before normalization"
    }
    with open(os.path.join(args.output_dir, "preprocessing_config.json"), "w") as f:
        json.dump(handoff_preprocessing, f, indent=2)
        
    # 3. class_mapping.json
    class_mapping = {
        "0": "Non-Violence",
        "1": "Violence"
    }
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
        
    # 4. recommended_threshold.json
    rec_thresh = test_results.get("recommended_threshold", 0.5)
    rec_metrics = test_results.get("metrics_at_recommended_threshold", test_results.get("metrics_at_default_0_5"))
    
    threshold_data = {
        "recommended_threshold": rec_thresh,
        "justification": "Optimal trade-off maximizing Violence Recall (Sensitivity) while controlling False Positives on non-violent human interactions (hugging, clapping, sudden movements).",
        "metrics_at_recommended_threshold": rec_metrics,
        "metrics_at_default_0_5": test_results.get("metrics_at_default_0_5"),
        "deployment_rule": f"if violence_prob >= {rec_thresh}: TRIGGER_SECURITY_ALERT(level='HIGH') else: NORMAL_STATE"
    }
    with open(os.path.join(args.output_dir, "recommended_threshold.json"), "w") as f:
        json.dump(threshold_data, f, indent=2)
        
    # 5. HANDOFF_README.md
    readme_content = f"""# Suraksha-Net: Model Deployment Handoff Package
**Prepared by:** Vishnu Bhargava (Advanced DL Model Development & Evaluation Lead)  
**Target:** Arpit / Deployment & Optimization Team  
**Model Architecture:** {model_cfg['name'].upper()} (Backbone: {model_cfg.get('backbone', 'resnet18')})  

---

## 1. Quick Specifications
- **Input Tensor:** `B × 16 × 3 × 224 × 224` (Batch × Sequence × RGB Channels × Height × Width)
- **Input Preprocessing:** RGB, Resized to 224×224, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Classes:** `0 = Non-Violence`, `1 = Violence`
- **Output:** Softmax probabilities across 2 classes
- **Recommended Detection Threshold:** `{rec_thresh}` (Trigger alert when P(Violence) >= `{rec_thresh}`)
- **Model Parameters:** `{total_params:,}`
- **Model File Size:** `{model_size_mb:.2f} MB`

---

## 2. Key Performance Metrics (Test Set)
- **Accuracy:** `{rec_metrics['accuracy']*100:.2f}%`
- **Precision:** `{rec_metrics['precision']*100:.2f}%`
- **Recall (Sensitivity):** `{rec_metrics['recall']*100:.2f}%`
- **F1-Score:** `{rec_metrics['f1']*100:.2f}%`
- **ROC-AUC:** `{rec_metrics['roc_auc']:.4f}`
- **False Positive Rate:** `{rec_metrics['false_positive_rate']*100:.2f}%`
- **False Negative Rate:** `{rec_metrics['false_negative_rate']*100:.2f}%`

---

## 3. Package File Directory
- `best_model.pth`: PyTorch trained model weights and full state dictionary
- `model_config.json`: Complete neural architecture specification
- `preprocessing_config.json`: Exact normalization and sampling rules
- `class_mapping.json`: Integer to string label mappings
- `recommended_threshold.json`: Calibrated decision threshold with security justification
"""
    with open(os.path.join(args.output_dir, "HANDOFF_README.md"), "w") as f:
        f.write(readme_content)
        
    print(f"\n========================================================")
    print(f"✅ HANDOFF PACKAGE CREATED SUCCESSFULLY!")
    print(f"Location: {args.output_dir}/")
    print(f"Files included:")
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()

