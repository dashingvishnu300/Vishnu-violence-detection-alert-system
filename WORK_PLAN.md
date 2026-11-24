# Suraksha-Net: Real-Time Violence Detection System
## Module: Advanced Deep Learning Model Development, Temporal Modeling & Evaluation
**Lead Researcher:** Vishnu Bhargava  
**Target Recipient / Deployment Lead:** Arpit  
**Supervision / Team:** Savitoj & Team  

---

## 1. System Architecture
```
Video / CCTV Stream (RTSP / Local Camera)
           │
           ▼
[Vishnu's DL Module]
  ├── Frame Sequence Generator (16 frames, 224x224 RGB, ImageNet Normalized)
  ├── Backbone Feature Extractor (ResNet18 / EfficientNet-B0 / VGG16)
  ├── Temporal Modeling (LSTM, hidden_size=256, dropout=0.3)
  └── Classification Head (Softmax: 0=Non-Violence, 1=Violence)
           │
           ▼ (best_model.pth + model_config.json + recommended_threshold.json)
[Arpit's Downstream Pipeline]
  ├── Model Optimization (ONNX / TensorRT / FP16)
  ├── Backend API / Streaming Engine
  ├── Android Mobile App
  └── Instant Security Alerts
```

---

## 2. Directory Structure
```
/home/arnav/Documents/vishnu project/
├── .venv/                         # Python 3.12 environment with CUDA PyTorch
├── configs/
│   └── default.yaml               # Centralized hyperparameters & data paths
├── data/
│   ├── rwf/                       # RWF-2000 dataset (Surveillance action)
│   ├── avd/                       # AIRTLab Violence Dataset (Hard negative human actions)
│   └── sfd/                       # Surveillance Fight Dataset (CCTV style)
├── datasets/
│   ├── video_dataset.py           # Custom PyTorch Dataset/DataLoader (B x T x C x H x W)
│   ├── audit_datasets.py          # Automated dataset health, FPS, corruption scanner
│   └── generate_mock_cctv.py      # Synthetic CCTV generator for verification dry-runs
├── models/
│   └── architectures.py           # VGG16, EfficientNet-B0, CNN-LSTM (ResNet18/VGG/EffNet)
├── training/
│   └── trainer.py                 # Multi-epoch trainer, AMP, Early Stopping, Checkpointing
├── evaluation/
│   ├── evaluator.py               # Precision, Recall, F1, ROC-AUC, FPR/FNR, Confusion Matrix
│   └── gradcam.py                 # Grad-CAM heatmap visualization & explainability
├── checkpoints/                   # Auto-saved epoch & best-checkpoint weights
├── experiments/                   # Experiment tracking (logs, loss curves, metrics.csv)
├── results/                       # Cross-dataset generalization matrices & comparisons
├── train.py                       # Main CLI training script
├── evaluate.py                    # Standalone checkpoint evaluator
├── cross_dataset_test.py          # Generalization matrix generator across datasets
├── compare_models.py              # Multi-model benchmarking table generator
├── export_handoff.py              # Packages all artifacts for Arpit
└── requirements.txt               # Frozen dependencies
```

---

## 3. Planned vs. Actual Execution Matrix

| Step | Planned Task | Actual Status & Implementation | Status |
| :--- | :--- | :--- | :---: |
| **Step 1** | GPU & CUDA Setup | NVIDIA RTX 5050 Laptop GPU (8GB, sm_120 Blackwell), PyTorch 2.14.1+cu130, CUDA 13.1. | **Completed ✅** |
| **Step 2** | Dataset Acquisition & Verification | • Purged 48 mock synthetic clips.<br>• Extracted full 1,976 real videos of RWF-2000 via standalone 7zz.<br>• Implemented scene-level splitting (SFD source mapping + AIRTLab paired cam1/cam2 + RWF source video clustering) yielding zero scene leakage across 2,626 total videos.<br>• Saved split lists (`train_clips.csv`, `val_clips.csv`, `test_clips.csv`). | **Completed ✅** |
| **Step 3** | Train VGG16 Baselines | Trained on SFD, AVD, and Combined (16 frames) with ImageNet initialization. Comparison table updated with backbone labeled as `vgg16`. | **Completed ✅** |
| **Step 4** | Train EfficientNet Baselines | Trained on SFD and Combined (16 frames) with ImageNet initialization. Comparison table updated with backbone labeled as `efficientnet_b0`. | **Completed ✅** |
| **Step 5** | Train Main Model (CNN + LSTM) | ResNet18 + LSTM (16 frames) trained with early stopping on validation loss, partial backbone freezing (conv1 through layer2), and enhanced data augmentation to reduce overfitting. | **Completed ✅** |
| **Step 6** | Cross-Dataset Generalization | Domain transfer evaluated across datasets (SFD $\to$ AVD, AVD $\to$ SFD, Combined $\to$ SFD/AVD/RWF). | **Completed ✅** |
| **Step 7** | Model Comparison & Master Matrix | Master benchmarking comparison table generated across all models, architectures, backbones, and metrics in `results/model_comparison_table.csv`. | **Completed ✅** |
| **Step 8** | Deployment Package for Arpit | Packaged clean weights-only checkpoint (`best_model.pth`, ~46.7 MB without optimizer state), CSV manifests, frozen threshold (`0.30`), and tested CLI in `handoff_to_arpit/`. | **Completed ✅** |

---

## 4. Execution Details & Reproducibility Guide

