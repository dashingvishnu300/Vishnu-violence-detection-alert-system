"""
Suraksha-Net: Grad-CAM Explainability Module
============================================
Visualizes where CNN / CNN-LSTM models focus attention:
- Differentiates human interaction areas from background noise
- Highlights why false positives (e.g., hugging, dancing) or false negatives occur
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    """Computes Grad-CAM heatmaps for a CNN backbone."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a single frame or video sequence.
        input_tensor: (1, T, C, H, W) or (1, C, H, W)
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward()
        
        # Pooled gradients across channels
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        cam_np = cam.squeeze().cpu().numpy()
        
        # Resize to 224x224
        if len(cam_np.shape) == 3:  # sequence
            heatmaps = []
            for c in cam_np:
                h = cv2.resize(c, (224, 224))
                heatmaps.append(h)
            return np.array(heatmaps)
        else:
            return cv2.resize(cam_np, (224, 224))


def overlay_gradcam(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Superimposes Grad-CAM heatmap over original RGB image."""
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_rgb)
    return superimposed


def save_gradcam_comparison(
    original_frame: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    true_label: str,
    confidence: float,
    save_path: str
):
    """Saves side-by-side explainability figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_frame)
    axes[0].set_title(f"Original CCTV Frame\nTrue: {true_label}")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay Attention\nPred: {pred_label} ({confidence:.2%})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

