"""
Suraksha-Net: Grad-CAM Generator for Error & Interpretability Analysis
======================================================================
Produces Grad-CAM visualizations on real test samples:
1. Correct Violence (True Positive)
2. Correct Non-Violence (True Negative)
3. False Positive (e.g., rapid movement or hugging)
4. False Negative (e.g., small or occluded fight)
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.video_dataset import build_dataloaders
from models.architectures import build_model
from evaluation.gradcam import GradCAM, overlay_gradcam, save_gradcam_comparison


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/combined_cnn_lstm_16frames/best_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "experiments/sfd_cnn_lstm_16frames/best_model.pth"
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Target layer: last conv block of ResNet18 (index 7)
    target_layer = model.cnn[7]
    grad_cam = GradCAM(model, target_layer)
    
    _, _, test_loader, _ = build_dataloaders(config)
    
    out_dir = "gradcam_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    found_cases = {}
    
    print("Searching test set for interpretability cases...")
    with torch.no_grad():
        for videos, labels, names, srcs in test_loader:
            v_dev = videos.to(device)
            outputs = model(v_dev)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= 0.5).int().cpu().numpy()
            labels_np = labels.numpy()
            
            for i in range(len(labels_np)):
                act = labels_np[i]
                prd = preds[i]
                prob = float(probs[i].item())
                
                # Middle frame (frame 8 of 16)
                frame_tensor = videos[i, 8]  # (3, 224, 224)
                # Unnormalize for visualization
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img = (frame_tensor.numpy() * std + mean).clip(0, 1)
                img_rgb = np.uint8(img.transpose(1, 2, 0) * 255)
                
                vid_sample = videos[i:i+1].to(device)
                
                if act == 1 and prd == 1 and "correct_violence" not in found_cases:
                    found_cases["correct_violence"] = (vid_sample, img_rgb, prob, names[i], act, prd)
                elif act == 0 and prd == 0 and "correct_nonviolence" not in found_cases:
                    found_cases["correct_nonviolence"] = (vid_sample, img_rgb, prob, names[i], act, prd)
                elif act == 0 and prd == 1 and "false_positive" not in found_cases:
                    found_cases["false_positive"] = (vid_sample, img_rgb, prob, names[i], act, prd)
                elif act == 1 and prd == 0 and "false_negative" not in found_cases:
                    found_cases["false_negative"] = (vid_sample, img_rgb, prob, names[i], act, prd)
                    
            if len(found_cases) == 4:
                break
                
    # Generate and save CAMs
    for case_name, (vid_tensor, rgb_frame, prob, vid_name, act, prd) in found_cases.items():
        vid_tensor.requires_grad = True
        model.zero_grad()
        
        # Forward and backward pass with cuDNN disabled for RNN backward compatibility
        with torch.backends.cudnn.flags(enabled=False):
            output = model(vid_tensor)
            score = output[:, 1].sum()
            score.backward()
        
        # Calculate pooled gradients across channels: (B*T, 1, H_feat, W_feat)
        weights = torch.mean(grad_cam.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * grad_cam.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        # Middle frame CAM (index 8 of 16 frames)
        if cam.shape[0] >= 16:
            cam_frame = cam[8, 0].detach().cpu().numpy()
        else:
            cam_frame = cam[-1, 0].detach().cpu().numpy()
            
        cam_resized = cv2.resize(cam_frame, (224, 224))
        overlay = overlay_gradcam(rgb_frame, cam_resized, alpha=0.5)
        
        save_path = os.path.join(out_dir, f"{case_name}.png")
        true_str = "Violence" if act == 1 else "Non-Violence"
        pred_str = "Violence" if prd == 1 else "Non-Violence"
        
        save_gradcam_comparison(
            original_frame=rgb_frame,
            heatmap=cam_resized,
            overlay=overlay,
            pred_label=pred_str,
            true_label=true_str,
            confidence=prob,
            save_path=save_path
        )
        print(f"✅ Saved Grad-CAM: {save_path} ({vid_name})")
        
    print(f"\nAll Grad-CAM interpretability artifacts saved to: {out_dir}/")


if __name__ == "__main__":
    main()
