"""
Suraksha-Net: Real-Time Standalone Inference Script for Deployment
=================================================================
Packaged for Arpit / Deployment & Optimization Team.

Usage:
  python run_inference.py --video path/to/cctv_video.mp4
  python run_inference.py --video path/to/cctv_video.mp4 --threshold 0.30 --device cuda
"""

import os
import sys
import json
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMInference(nn.Module):
    """Standalone ResNet18 + LSTM architecture for inference without external dependencies."""
    def __init__(self, num_classes=2, lstm_hidden_size=256, lstm_num_layers=1):
        super().__init__()
        base_resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(base_resnet.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)  # (B*T, 512, 1, 1)
        feat = feat.view(B, T, 512)
        lstm_out, _ = self.lstm(feat)
        out = lstm_out[:, -1, :]  # Last time step
        logits = self.classifier(out)
        return logits


def preprocess_video(video_path: str, seq_len: int = 16, img_size: int = 224) -> Tuple[torch.Tensor, int]:
    """Extract frames uniformly, resize, convert RGB, and normalize."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"No valid frames read from {video_path}")

    # Uniform sampling
    if n_frames >= seq_len:
        indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        sampled = list(frames)
        while len(sampled) < seq_len:
            sampled.append(frames[-1])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    processed = []
    for f in sampled:
        resized = cv2.resize(f, (img_size, img_size))
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std
        # (H, W, C) -> (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        processed.append(transposed)

    # (T, C, H, W) -> (1, T, C, H, W)
    tensor = torch.FloatTensor(np.array(processed)).unsqueeze(0)
    return tensor, n_frames


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load model weights."""
    model = CNNLSTMInference(num_classes=2, lstm_hidden_size=256, lstm_num_layers=1)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_weights = os.path.join(script_dir, "best_model.pth")
    if not os.path.isfile(default_weights) and os.path.isfile("best_model.pth"):
        default_weights = "best_model.pth"

    parser = argparse.ArgumentParser(description="Suraksha-Net Inference Runner")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4, .avi)")
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.30, help="Alert threshold (default: 0.30 from validation tuning)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load Model
    model = load_model(args.weights, device)

    # Preprocess Video
    video_tensor, total_frames = preprocess_video(args.video, seq_len=16, img_size=224)
    video_tensor = video_tensor.to(device)

    # Predict
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_violence = float(probs[1])
    prob_nonviolence = float(probs[0])
    alert_triggered = bool(prob_violence >= args.threshold)
    label = "Violence" if alert_triggered else "Non-Violence"

    result = {
        "status": "success",
        "video": os.path.abspath(args.video),
        "total_video_frames": total_frames,
        "prediction": label,
        "violence_probability": round(prob_violence, 4),
        "non_violence_probability": round(prob_nonviolence, 4),
        "alert_triggered": alert_triggered,
        "alert_threshold": args.threshold,
        "device_used": str(device)
    }

    print("\n" + "="*50)
    print("SURAKSHA-NET DEPLOYMENT INFERENCE RESULT:")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

