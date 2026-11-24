"""
Suraksha-Net: Training Engine
==============================
Handles training loop, validation, checkpointing, early stopping, AMP.
"""

import os
import sys
import time
import json
import logging
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Complete training pipeline with AMP, checkpointing, early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        experiment_dir: str,
        class_weights=None,
    ):
        self.config = config
        cfg_tr = config["training"]
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Loss function
        if cfg_tr.get("use_class_weights", False) and class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"  Using weighted loss: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        opt_name = cfg_tr.get("optimizer", "adamw").lower()
        lr = cfg_tr.get("learning_rate", 1e-4)
        wd = cfg_tr.get("weight_decay", 1e-4)
        
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, momentum=cfg_tr.get("momentum", 0.9), weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Scheduler
        sched_name = cfg_tr.get("scheduler", "cosine").lower()
        if sched_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min",
                patience=cfg_tr.get("scheduler_patience", 5),
                factor=0.5, verbose=True)
        elif sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg_tr.get("epochs", 30))
        elif sched_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=cfg_tr.get("scheduler_step_size", 10),
                gamma=cfg_tr.get("scheduler_gamma", 0.1))
        else:
            self.scheduler = None
        
        # AMP
        self.use_amp = cfg_tr.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping on validation loss (Requirement 5 & 6)
        self.early_stopping = EarlyStopping(
            patience=cfg_tr.get("early_stopping_patience", 7),
            mode="min"  # monitoring validation loss to prevent overfitting
        )
        
        # Training state
        self.epochs = cfg_tr.get("epochs", 30)
        self.gradient_clip = cfg_tr.get("gradient_clip", 1.0)
        self.log_interval = config["experiment"].get("log_interval", 10)
        self.start_epoch = 0
        self.best_f1 = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.history = []
        
        # Metrics CSV
        self.metrics_path = os.path.join(experiment_dir, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "val_precision", "val_recall", "val_f1", "lr", "time_sec"
            ])
        
        # Training log
        self.log_path = os.path.join(experiment_dir, "training.log")
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint. Best model saved as clean weights-only (~46 MB)."""
        if is_best:
            best_path = os.path.join(self.experiment_dir, "best_model.pth")
            clean_checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
                "best_f1": self.best_f1,
                "config": self.config,
                "history": self.history,
            }
            torch.save(clean_checkpoint, best_path)
            print(f"  💾 Clean best model saved (Val Loss={self.best_val_loss:.4f}, F1={self.best_f1:.4f})")
    
    def load_checkpoint(self, path):
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_f1 = checkpoint.get("best_f1", 0.0)
        self.history = checkpoint.get("history", [])
        print(f"Resumed from epoch {self.start_epoch}, best F1={self.best_f1:.4f}")
    
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                     leave=False)
        
        for batch_idx, (videos, labels, names, sources) in enumerate(pbar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.1f}%"
            })
        
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate and compute metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
                     leave=False)
        
        for videos, labels, names, sources in pbar:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * videos.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        preds = np.array(all_preds)
        labels_np = np.array(all_labels)
        
        avg_loss = running_loss / len(labels_np)
        accuracy = accuracy_score(labels_np, preds)
        precision = precision_score(labels_np, preds, zero_division=0)
        recall = recall_score(labels_np, preds, zero_division=0)
        f1 = f1_score(labels_np, preds, zero_division=0)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": preds,
            "labels": labels_np,
            "probabilities": np.array(all_probs),
        }
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config['experiment']['name']}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Optimizer: {self.config['training']['optimizer']}")
        print(f"  LR: {self.config['training']['learning_rate']}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            
            # Log
            log_str = (
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f} "
                f"F1: {val_metrics['f1']:.4f} | "
                f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
            )
            print(log_str)
            logger.info(log_str)
            
            # Save metrics
            with open(self.metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                    f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}",
                    f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}",
                    f"{val_metrics['f1']:.6f}", f"{lr:.8f}", f"{epoch_time:.1f}"
                ])
            
            # History
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": lr,
                "time": epoch_time,
            })
            
            # Check for best model based on validation loss (with F1 tracking)
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_f1 = val_metrics["f1"]
                self.best_epoch = epoch + 1
            
            # Save checkpoint (best model saved as clean weights-only)
            if is_best or epoch == self.epochs - 1:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Early stopping on validation loss
            if self.early_stopping(val_metrics["loss"]):
                print(f"\n⚠️  Early stopping at epoch {epoch+1} "
                      f"(validation loss did not improve for {self.early_stopping.patience} epochs)")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        print(f"  Model saved: {self.experiment_dir}/best_model.pth")
        print(f"{'='*60}\n")
        
        return self.history

