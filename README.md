# Suraksha-Net: Surveillance Video Violence Detection
## Module: Deep Learning Model Development, Temporal Modeling & Evaluation

**Module Lead:** Vishnu Bhargava  
**Collaborators:** Arpit (Deployment & Edge Optimization), Savitoj, Ojas  
**Institution:** Major Project / Final Year Engineering  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0%20(RTX%205050)-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**Suraksha-Net** is an automated video surveillance violence detection system designed to monitor CCTV video clips, detect physical violence or altercations with low latency, and trigger security alerts to on-duty personnel via an Android mobile client.

This repository houses the complete deep learning core developed by **Vishnu Bhargava**, encompassing dataset auditing, scene-level stratified data splitting, baseline 2D CNNs, the champion **CNN + LSTM** temporal model, cross-dataset transfer evaluation, Grad-CAM explainability, and the deployment handoff package.

---

## 🚀 Key Experimental Results

All experiments were conducted on a local **NVIDIA GeForce RTX 5050 Laptop GPU (8 GB GDDR6, sm_120 Blackwell)** using Automatic Mixed Precision (AMP FP16). All comparative models are trained with identical sequence length (16 frames) and early stopping on validation loss.

| Model Architecture | Dataset | Sequence Length | Parameters | Test Accuracy | Precision | Violence Recall | F1-Score | ROC-AUC | False Positive Rate |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **VGG16 Baseline** | SFD | 16 frames | 121.7M | 60.42% | 55.81% | **100.00%** | 71.64% | 0.7969 | 79.17% |
| **EfficientNet-B0** | SFD | 16 frames | **4.66M** | 68.75% | 62.16% | 95.83% | 75.41% | 0.7674 | 58.33% |
| **CNN + LSTM** | SFD | 16 frames | 12.00M | 72.92% | 72.00% | 75.00% | 73.47% | 0.7969 | 29.17% |
| **CNN + LSTM (Champion)** | **Combined (SFD + AVD + RWF)** | 16 frames | 12.00M | **83.00%** | **82.81%** | **89.83%** | **86.18%** | **0.8876** | **21.95%** |

> **Operational Alert Threshold:**  
> The alert threshold is calibrated strictly on validation data to optimize the operational balance and frozen before evaluation on untouched test videos.

---

## 🧠 Scientific Highlights & Key Observations

1. **Mitigating False Alarms with Temporal Modeling:**  
   Static 2D CNNs (e.g. VGG16) evaluate frames in isolation, mistaking innocent fast movements (hugging, dancing, clapping, waving) for violence, resulting in high false positive rates. The **CNN + LSTM** models temporal motion across 16 frames, substantially reducing false alarms.
2. **Scene-Level Data Partitioning:**  
   Splits are enforced at the scene and source-video level rather than per-clip, ensuring that multiple camera views of the same scene and multiple cuts from the same source video remain in the same split.
3. **Cross-Dataset Generalization:**  
   Domain transfer tests between outdoor CCTV (SFD), multi-camera indoor rooms (AIRTLab/AVD), and YouTube surveillance footage (RWF-2000) quantify the domain gap across surveillance environments.
4. **Grad-CAM Interpretability:**  
   Gradient-weighted Class Activation Mapping confirms the network focuses on interacting human limbs and grappling contact points rather than static background scenery.

---

## 📂 Repository Structure

```
.
├── datasets/
│   ├── video_dataset.py          # PyTorch VideoDataset & DataLoader (16 frames, BxTxCxHxW)
│   ├── prepare_datasets.py        # Dataset structure parser (SFD, AVD, Combined)
│   └── audit_datasets.py          # Video resolution, duration, FPS, and integrity audit
├── models/
│   └── architectures.py          # VGG16, EfficientNet-B0, ResNet18+LSTM, R3D-18
├── training/
│   └── trainer.py                # AMP FP16 trainer with early stopping & Cosine LR
├── evaluation/
│   ├── evaluator.py              # Threshold sweeping, ROC-AUC, confusion matrix
│   ├── gradcam.py                # Spatio-temporal Grad-CAM implementation
│   └── run_gradcam_analysis.py   # Explainability pipeline on test samples
├── handoff_to_arpit/             # Official deployment package for Arpit
│   ├── model_config.json         # Architecture specs for ONNX / TensorRT
│   ├── preprocessing_config.json # Sampling and ImageNet normalization rules
│   ├── class_mapping.json        # 0 = Non-Violence, 1 = Violence
│   ├── recommended_threshold.json# Calibrated 0.40 threshold justification
│   ├── HANDOFF_README.md         # Deployment & inference documentation
│   └── gradcam/                  # TP, TN, FP, FN visual maps
├── results/                      # Benchmarking CSVs, metrics, and logs
│   ├── model_comparison_table.csv# Master metrics table
│   └── FINAL_RESULTS_REPORT.md   # Comprehensive project report
├── VISHNU_WORK_COMPREHENSIVE_REPORT.pdf  # 12-page Defense PDF Report
├── VISHNU_WORK_COMPREHENSIVE_REPORT.md   # Academic Report Markdown Source
├── train.py                      # Multi-model training entry point
├── evaluate.py                   # Standalone evaluation entry point
├── cross_dataset_test.py         # Zero-shot domain transfer script
├── compare_models.py             # Automated benchmarking aggregator
├── export_handoff.py             # Packaging script for deployment team
└── requirements.txt              # Environment dependencies
```

---

## 🛠️ Quickstart & Setup

### 1. Installation
```bash
git clone https://github.com/dashingvishnu300/Vishnu-violence-detection-alert-system.git
cd vishnu_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Training the Champion Model (CNN + LSTM)
```bash
python train.py --dataset combined --model cnn_lstm --epochs 8 --batch-size 8 --lr 1e-4 --amp
```

### 3. Running Comprehensive Evaluation
```bash
python evaluate.py --model-dir experiments/combined_cnn_lstm_16frames --dataset combined
```

### 4. Running Grad-CAM Explainability
```bash
python evaluation/run_gradcam_analysis.py
```

### 5. Running Quick Inference
```python
import torch, json
from models.architectures import build_model

# Load architecture and weights
with open("handoff_to_arpit/model_config.json") as f:
    config = json.load(f)
model = build_model(config)
ckpt = torch.load("handoff_to_arpit/best_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Input video tensor: (Batch, 16 frames, 3 RGB channels, 224 H, 224 W)
video = torch.randn(1, 16, 3, 224, 224)

with torch.no_grad():
    prob_violence = torch.softmax(model(video), dim=1)[0, 1].item()

is_alert = prob_violence >= 0.40  # Recommended calibrated threshold
print(f"P(Violence): {prob_violence:.3f} | Security Alert: {is_alert}")
```

---

## 📄 Academic Defense Documentation

A comprehensive **12-page Project Defense Report** is included in this repository:
- **[`VISHNU_WORK_COMPREHENSIVE_REPORT.pdf`](VISHNU_WORK_COMPREHENSIVE_REPORT.pdf)**

It includes mathematical formulations, training dynamics, ROC curves, confusion matrices, Grad-CAM visualizations, and viva defense Q&A.

---

## 👥 Authors & Team
- **Vishnu Bhargava** — Deep Learning Modeling, Temporal Architecture & Evaluation Lead
- **Ojas** — Team Lead & Project Architecture
- **Arpit** — Downstream Edge Optimization, ONNX/TensorRT & Android Integration Lead
- **Savitoj** — Dataset Sourcing & Preparation Support

