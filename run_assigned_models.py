"""
Suraksha-Net: Run Assigned Missing Training Experiments
======================================================
1. combined_vgg16_16frames (VGG16 on Combined 650 videos)
2. combined_efficientnet_16frames (EfficientNet on Combined 650 videos)
3. combined_cnn_lstm_16frames (Clean champion retrained on verified 650 videos)
"""

import subprocess
import sys
import os

runs = [
    {
        "name": "combined_vgg16_16frames",
        "cmd": [sys.executable, "train.py", "--dataset", "combined", "--model", "vgg16", "--epochs", "5", "--batch-size", "8", "--exp-name", "combined_vgg16_16frames"]
    },
    {
        "name": "combined_efficientnet_16frames",
        "cmd": [sys.executable, "train.py", "--dataset", "combined", "--model", "efficientnet", "--epochs", "5", "--batch-size", "8", "--exp-name", "combined_efficientnet_16frames"]
    },
    {
        "name": "combined_cnn_lstm_16frames",
        "cmd": [sys.executable, "train.py", "--dataset", "combined", "--model", "cnn_lstm", "--backbone", "resnet18", "--epochs", "5", "--batch-size", "8", "--exp-name", "combined_cnn_lstm_16frames"]
    }
]

for r in runs:
    print(f"\n{'='*70}")
    print(f"STARTING ASSIGNED RUN: {r['name']}")
    print(f"{'='*70}\n")
    ret = subprocess.run(r["cmd"])
    if ret.returncode != 0:
        print(f"❌ Failed: {r['name']}")
    else:
        print(f"✅ Finished: {r['name']}")

