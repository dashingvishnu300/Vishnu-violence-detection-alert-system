# Suraksha-Net: Model Deployment Handoff Package
**Prepared by:** Vishnu Bhargava (Advanced DL Model Development & Evaluation Lead)  
**Target:** Arpit / Deployment & Optimization Team  
**Model Architecture:** CNN_LSTM (ResNet18 Backbone + LSTM Temporal Aggregation)  

---

## 1. Quick Start (Single Tested Command)

Run inference on any real CCTV video clip with one command:

```bash
# 1. Run inference on a video (automatically utilizes CUDA if available):
python run_inference.py --video path/to/cctv_clip.mp4

# 2. Specify custom alert threshold or device:
python run_inference.py --video path/to/cctv_clip.mp4 --threshold 0.30 --device cuda
```

### Example Real Output (Non-Violent Activity Clip):
```json
{
  "status": "success",
  "video": "/home/arnav/Documents/vishnu project/data/avd/NonViolence/avd_nonviolent_cam2_1.mp4",
  "total_video_frames": 158,
  "prediction": "Non-Violence",
  "violence_probability": 0.0388,
  "non_violence_probability": 0.9612,
  "alert_triggered": false,
  "alert_threshold": 0.30,
  "device_used": "cuda"
}
```

### Example Real Output (Violent Altercation Clip):
```json
{
  "status": "success",
  "video": "/home/arnav/Documents/vishnu project/data/sfd/Violence/fi001.mp4",
  "total_video_frames": 64,
  "prediction": "Violence",
  "violence_probability": 0.9424,
  "non_violence_probability": 0.0576,
  "alert_triggered": true,
  "alert_threshold": 0.30,
  "device_used": "cuda"
}
```

---

## 2. Input Specifications & Preprocessing Pipeline
- **Input Tensor Format:** `[B, 16, 3, 224, 224]` (Batch × Sequence × Channels × Height × Width)
- **Sequence Length:** 16 frames uniformly sampled across video duration.
- **Spatial Dimensions:** 224 × 224 pixels.
- **Color Space:** RGB (`cv2.COLOR_BGR2RGB`).
- **Normalization:** ImageNet statistics:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Scaling:** Pixels scaled to `[0.0, 1.0]` before normalization.

---

## 3. Class Mapping & Decision Threshold
- **Class Mapping:**
  - `0`: **Non-Violence** (Normal human activity, walking, loitering, conversation)
  - `1`: **Violence** (Physical fighting, assault, aggressive altercations)
- **Calibrated Alert Threshold:** **`0.30`**
  - Tuned on the validation set to maximize violence sensitivity (recall).
  - An event triggers an active security alert whenever `P(Violence) >= 0.30`.

---

## 4. Final Benchmark Performance Metrics (Held-Out Test Set)

Evaluated on untouched test videos across all three datasets (510 total surveillance clips: 400 RWF-2000, 54 AIRTLab/AVD, 56 SFD):

### Overall Combined Metrics (At Frozen Threshold $\tau = 0.30$):

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Accuracy** | **74.12%** | Overall correct classifications across all environments |
| **Precision** | **69.03%** | Purity of generated high-priority alarms |
| **Violence Recall (Sensitivity)** | **91.35%** | Caught 243 out of 266 real violent altercations |
| **F1-Score** | **78.64%** | Harmonic balance between alarm purity and safety critical detection |
| **ROC-AUC** | **0.8203** | Discriminative separation across thresholds |
| **False Alarms (FP)** | 109 clips | Non-violent clips flagged due to rapid motion/running |
| **Missed Fights (FN)** | 23 clips | Subtle scuffles or low-contrast night conditions |

### Per-Dataset Performance Breakdown:

| Dataset | Test Clips | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | False Alarms | Missed Fights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AIRTLab (AVD)** | 54 | 74.07% | 72.00% | **100.00%** | 83.72% | 0.9059 | 14 | **0** |
| **RWF-2000** | 400 | 75.75% | 70.52% | 88.50% | 78.49% | 0.8318 | 74 | 23 |
| **SFD** | 56 | 62.50% | 58.82% | **100.00%** | 74.07% | 0.7962 | 21 | **0** |

---

## 5. File Manifest

### Model & Runtime Code:
- `best_model.pth`: Clean PyTorch state dictionary (`46 MB`, weights-only without optimizer state)
- `run_inference.py`: Production-ready inference runner (loads weights, processes video, returns JSON)
- `model_config.json`: Architecture specifications (ResNet18, 512 feat dim, 256 LSTM hidden, 1 layer)
- `preprocessing_config.json`: Normalization mean/std, tensor dimensions, and frame sampling parameters
- `class_mapping.json`: Integer to string class mapping dictionary (`0 = Non-Violence`, `1 = Violence`)

### Verification & Evaluation Results:
- `test_results.json`: Full test evaluation output, overall and per-dataset metrics, and validation sweep
- `recommended_threshold.json`: Calibrated validation alert threshold (`0.30`) with deployment rule and metrics
- `error_analysis.json`: Detailed review of false alarms (FP) and missed fights (FN) with root-cause explanations
- `model_comparison_table.csv`: Master benchmark comparison table across VGG16, EfficientNet, and CNN-LSTM
- `generalization_matrix.csv`: Cross-dataset zero-shot transfer results (Combined $\to$ RWF, AVD, SFD)
- `training_metrics.csv`: Per-epoch convergence log (train/val loss, accuracy, precision, recall, F1, lr)
- `confusion_matrix.png` & `roc_curve.png`: Evaluation diagnostic plots at the frozen threshold
- `loss_curve.png` & `accuracy_f1_curve.png`: Training convergence and validation dynamics
- `gradcam/`: Visual heatmaps for true positive, true negative, false positive, and false negative predictions

### Data Manifests & Documentation:
- `train_clips.csv`: Complete training split manifest (1,801 clips across SFD, AVD, and RWF)
- `val_clips.csv`: Complete validation split manifest (315 clips)
- `test_clips.csv`: Complete held-out test split manifest (510 clips)
- `VISHNU_WORK_COMPREHENSIVE_REPORT.pdf`: Complete 12-page defense report and experimental documentation
