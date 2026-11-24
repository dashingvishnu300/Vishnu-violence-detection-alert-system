"""
Suraksha-Net: Training Entry Point
==================================
Usage:
  python train.py --dataset rwf --model cnn_lstm
  python train.py --dataset combined --model vgg16
  python train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import json
import torch

from datasets.video_dataset import build_dataloaders
from models.architectures import build_model
from training.trainer import Trainer
from evaluation.evaluator import evaluate_model, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Suraksha-Net Video Violence Detection Trainer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, choices=["rwf", "avd", "sfd", "combined"], help="Dataset override")
    parser.add_argument("--model", type=str, default=None, choices=["vgg16", "efficientnet", "cnn_lstm", "r3d"], help="Model architecture override")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone for CNN-LSTM (resnet18, efficientnet_b0, etc.)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--seq-len", type=int, default=None, choices=[8, 16, 32], help="Sequence length override")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze CNN backbone feature layers")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Unfreeze all layers")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load base config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Apply CLI overrides
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.model:
        config["model"]["name"] = args.model
        # VGG16 is heavy, default freeze backbone for baseline
        if args.model == "vgg16":
            config["model"]["freeze_backbone"] = True
    if args.freeze_backbone:
        config["model"]["freeze_backbone"] = True
    if args.unfreeze_backbone:
        config["model"]["freeze_backbone"] = False
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.seq_len:
        config["dataset"]["sequence_length"] = args.seq_len
        
    # Experiment directory
    ds_name = config["dataset"]["name"]
    mdl_name = config["model"]["name"]
    seq_len = config["dataset"]["sequence_length"]
    
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{ds_name}_{mdl_name}_{seq_len}frames"
        
    config["experiment"]["name"] = exp_name
    exp_dir = os.path.join(config["experiment"].get("save_dir", "experiments"), exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save active config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\n========================================================")
    print(f"SURAKSHA-NET EXPERIMENT: {exp_name}")
    print(f"Dataset: {ds_name} | Model: {mdl_name} | Seq Len: {seq_len}")
    print(f"Target Output Directory: {exp_dir}")
    print(f"========================================================\n")
    
    # Build DataLoaders
    try:
        train_loader, val_loader, test_loader, ds_info = build_dataloaders(config)
    except FileNotFoundError as e:
        print(f"\n❌ DATASET ERROR: {e}")
        print("\nPlease ensure dataset videos are placed in the data/ folder.")
        sys.exit(1)
        
    print(f"Dataset Summary:")
    print(f"  Train: {ds_info['train_count']} videos ({ds_info['train_violence']} Violence, {ds_info['train_nonviolence']} Non-Violence)")
    print(f"  Val:   {ds_info['val_count']} videos ({ds_info['val_violence']} Violence, {ds_info['val_nonviolence']} Non-Violence)")
    print(f"  Test:  {ds_info['test_count']} videos ({ds_info['test_violence']} Violence, {ds_info['test_nonviolence']} Non-Violence)")
    
    # Build Model
    model = build_model(config)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=exp_dir,
        class_weights=ds_info["class_weights"] if config["training"].get("use_class_weights") else None
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Run Training
    history = trainer.train()
    
    # Generate Training Curves
    plot_learning_curves(os.path.join(exp_dir, "metrics.csv"), exp_dir)
    
    # Run Final Test Set Evaluation
    print("\nRunning final evaluation on held-out test set...")
    best_ckpt_path = os.path.join(exp_dir, "best_model.pth")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=trainer.device,
        save_dir=exp_dir,
        val_loader=val_loader,
        thresholds=config["evaluation"].get("thresholds", [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    )
    
    m_frozen = eval_results.get("metrics_at_frozen_threshold", eval_results["metrics_at_default_0_5"])
    frozen_th = eval_results.get("frozen_validation_threshold", 0.50)
    print(f"\n========================================================")
    print(f"OVERALL TEST RESULTS (Evaluated at Frozen Val Threshold = {frozen_th:.2f}):")
    print(f"  Accuracy:  {m_frozen['accuracy']*100:.2f}%")
    print(f"  Precision: {m_frozen['precision']*100:.2f}%")
    print(f"  Recall:    {m_frozen['recall']*100:.2f}% (Violence Sensitivity)")
    print(f"  F1-Score:  {m_frozen['f1']*100:.2f}%")
    print(f"  False Positives (Alarms): {m_frozen['false_positives']}")
    print(f"  False Negatives (Missed): {m_frozen['false_negatives']}")
    print(f"  ROC-AUC:   {m_frozen['roc_auc']:.4f}")
    print(f"  Recommended Alert Threshold: {frozen_th:.2f}")

    per_ds = eval_results.get("per_dataset_metrics", {})
    if len(per_ds) > 1:
        print(f"\n--- PER-DATASET BREAKDOWN (At Frozen Threshold {frozen_th:.2f}) ---")
        for ds_k, ds_m in per_ds.items():
            print(f"  [{ds_k.upper():8s}] Acc: {ds_m['accuracy']*100:5.2f}% | Prec: {ds_m['precision']*100:5.2f}% | Rec: {ds_m['recall']*100:5.2f}% | F1: {ds_m['f1']*100:5.2f}% | AUC: {ds_m['roc_auc']:.4f} | FP: {ds_m['false_positives']:2d} | FN: {ds_m['false_negatives']:2d}")
    print(f"Full details saved in: {exp_dir}/test_results.json")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()

