"""
Suraksha-Net: Model Definitions
================================
VGG16 Baseline, EfficientNet-B0, CNN-LSTM, and optional R3D-18.
"""

import torch
import torch.nn as nn
from torchvision import models


# ==================================================================
# MODEL 1: VGG16 Baseline (single-frame classification)
# ==================================================================
class VGG16Model(nn.Module):
    """
    VGG16 with ImageNet pretrained weights.
    Modified classifier for binary violence detection.
    Treats each frame independently (no temporal modeling).
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vgg16(weights=weights)
        
        # Freeze feature layers
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.classifier[0].in_features  # 25088
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            logits: (B, num_classes) — averages predictions across frames
        """
        B, T, C, H, W = x.shape
        # Reshape to process all frames at once
        x = x.view(B * T, C, H, W)
        logits = self.backbone(x)  # (B*T, num_classes)
        # Average across frames
        logits = logits.view(B, T, -1).mean(dim=1)  # (B, num_classes)
        return logits
    
    def unfreeze_last_block(self):
        """Unfreeze the last conv block for fine-tuning."""
        for param in self.backbone.features[24:].parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ==================================================================
# MODEL 2: EfficientNet-B0 Lightweight
# ==================================================================
class EfficientNetModel(nn.Module):
    """
    EfficientNet-B0 with ImageNet pretrained weights.
    Lightweight alternative to VGG16.
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.backbone(x)
        logits = logits.view(B, T, -1).mean(dim=1)
        return logits
    
    def unfreeze_last_block(self):
        for param in list(self.backbone.features.children())[-2:]:
            for p in param.parameters():
                p.requires_grad = True
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# ==================================================================
# MODEL 3: CNN + LSTM (Main Temporal Model)
# ==================================================================
class CNNLSTM(nn.Module):
    """
    Main model: CNN feature extractor + LSTM temporal modeling.
    
    Pipeline:
        Video Frames → CNN (per-frame features) → LSTM → FC → Classification
    
    Supports multiple CNN backbones: resnet18, efficientnet_b0, mobilenetv3, vgg16
    """
    
    BACKBONE_CONFIGS = {
        "resnet18":       {"model_fn": lambda w: models.resnet18(weights=w),
                           "weights": models.ResNet18_Weights.DEFAULT,
                           "feature_dim": 512},
        "resnet50":       {"model_fn": lambda w: models.resnet50(weights=w),
                           "weights": models.ResNet50_Weights.DEFAULT,
                           "feature_dim": 2048},
        "efficientnet_b0": {"model_fn": lambda w: models.efficientnet_b0(weights=w),
                            "weights": models.EfficientNet_B0_Weights.DEFAULT,
                            "feature_dim": 1280},
        "mobilenetv3":    {"model_fn": lambda w: models.mobilenet_v3_small(weights=w),
                           "weights": models.MobileNet_V3_Small_Weights.DEFAULT,
                           "feature_dim": 576},
        "vgg16":          {"model_fn": lambda w: models.vgg16(weights=w),
                           "weights": models.VGG16_Weights.DEFAULT,
                           "feature_dim": 4096},
    }
    
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone}. "
                             f"Choose from {list(self.BACKBONE_CONFIGS.keys())}")
        
        cfg = self.BACKBONE_CONFIGS[backbone]
        self.feature_dim = cfg["feature_dim"]
        self.backbone_name = backbone
        
        # Build CNN feature extractor
        weights = cfg["weights"] if pretrained else None
        base_model = cfg["model_fn"](weights)
        
        if backbone.startswith("resnet"):
            # Remove final FC, keep avgpool
            children = list(base_model.children())[:-1]
            self.cnn = nn.Sequential(*children)
            # Partially freeze early layers (stem + layer1 + layer2) to prevent overfitting on CCTV
            if not freeze_backbone:
                for child in children[:6]:  # conv1, bn1, relu, maxpool, layer1, layer2
                    for param in child.parameters():
                        param.requires_grad = False
        elif backbone == "efficientnet_b0":
            self.cnn = nn.Sequential(
                base_model.features,
                base_model.avgpool,
            )
        elif backbone == "mobilenetv3":
            self.cnn = nn.Sequential(
                base_model.features,
                base_model.avgpool,
            )
        elif backbone == "vgg16":
            self.cnn = nn.Sequential(
                base_model.features,
                base_model.avgpool,
                nn.Flatten(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
            )
        
        # Freeze CNN if requested
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    
    def extract_features(self, x):
        """Extract CNN features from a batch of frames."""
        features = self.cnn(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Extract features for each frame
        x = x.view(B * T, C, H, W)
        features = self.extract_features(x)  # (B*T, feature_dim)
        features = features.view(B, T, -1)   # (B, T, feature_dim)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(features)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)
        
        # Classification
        logits = self.classifier(last_hidden)  # (B, num_classes)
        return logits
    
    def unfreeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = True


# ==================================================================
# MODEL 4 (Optional): R3D-18 (3D CNN)
# ==================================================================
class R3D18Model(nn.Module):
    """
    R3D-18 video classification model.
    Directly processes video clips with 3D convolutions.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.video.R3D_18_Weights.DEFAULT if pretrained else None
        self.model = models.video.r3d_18(weights=weights)
        
        # Replace final FC
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            logits: (B, num_classes)
        """
        # R3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


# ==================================================================
# Factory function
# ==================================================================
def build_model(config: dict) -> nn.Module:
    """Build model from config dict."""
    cfg = config["model"]
    name = cfg["name"]
    num_classes = cfg.get("num_classes", 2)
    pretrained = cfg.get("pretrained", True)
    freeze = cfg.get("freeze_backbone", False)
    
    if name == "vgg16":
        model = VGG16Model(num_classes, pretrained, freeze)
    elif name == "efficientnet":
        model = EfficientNetModel(num_classes, pretrained, freeze)
    elif name == "cnn_lstm":
        model = CNNLSTM(
            backbone=cfg.get("backbone", "resnet18"),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze,
            lstm_hidden_size=cfg.get("lstm_hidden_size", 256),
            lstm_num_layers=cfg.get("lstm_num_layers", 1),
            lstm_dropout=cfg.get("lstm_dropout", 0.3),
            classifier_dropout=cfg.get("classifier_dropout", 0.5),
        )
    elif name == "r3d":
        model = R3D18Model(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {name}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    return model

