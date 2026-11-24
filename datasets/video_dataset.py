"""
Suraksha-Net: Video Violence Detection Dataset
===============================================
Custom PyTorch Dataset for loading video clips as frame sequences.
Handles RWF-2000, AVD, SFD, and combined datasets.
"""

import os
import glob
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """
    Generic video dataset that loads video files and returns frame sequences.
    
    Returns:
        video_tensor: (T, C, H, W) tensor of frames
        label: 0 (NonViolence) or 1 (Violence)
        video_name: filename of the video
        dataset_source: which dataset this video came from
    """
    
    CLASS_MAP = {"nonviolence": 0, "nonfight": 0, "non-violence": 0, "non_violence": 0,
                 "violence": 1, "fight": 1}
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        sources: List[str],
        sequence_length: int = 16,
        image_size: int = 224,
        sampling_strategy: str = "uniform",
        transform=None,
        is_training: bool = False,
        normalize_mean: Tuple = (0.485, 0.456, 0.406),
        normalize_std: Tuple = (0.229, 0.224, 0.225),
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.sources = sources
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
        self.is_training = is_training
        
        # Build transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transforms(
                image_size, is_training, normalize_mean, normalize_std
            )
    
    def _build_transforms(self, image_size, is_training, mean, std):
        """Build train or eval transforms."""
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load all frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample exactly self.sequence_length frames from the video."""
        n_frames = len(frames)
        seq_len = self.sequence_length
        
        if n_frames == 0:
            # Return blank frames if video is empty/corrupt
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank] * seq_len
        
        if n_frames == seq_len:
            return frames
        
        if n_frames < seq_len:
            # Pad by repeating last frame
            padded = list(frames)
            while len(padded) < seq_len:
                padded.append(frames[-1])
            return padded
        
        # n_frames > seq_len: sample
        if self.sampling_strategy == "uniform":
            indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
        elif self.sampling_strategy == "consecutive":
            if self.is_training:
                start = random.randint(0, n_frames - seq_len)
            else:
                start = (n_frames - seq_len) // 2  # center crop for eval
            indices = list(range(start, start + seq_len))
        elif self.sampling_strategy == "random":
            if self.is_training:
                indices = sorted(random.sample(range(n_frames), seq_len))
            else:
                indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
        else:
            indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
        
        return [frames[i] for i in indices]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        source = self.sources[idx]
        video_name = os.path.basename(video_path)
        
        # Load and sample frames
        all_frames = self._load_video_frames(video_path)
        sampled_frames = self._sample_frames(all_frames)
        
        # Apply transforms to each frame
        transformed = []
        for frame in sampled_frames:
            t_frame = self.transform(frame)  # (C, H, W)
            transformed.append(t_frame)
        
        # Stack: (T, C, H, W)
        video_tensor = torch.stack(transformed, dim=0)
        
        return video_tensor, label, video_name, source
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse-frequency class weights for imbalanced data."""
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        weights = total / (len(counts) * counts.astype(float) + 1e-6)
        return torch.FloatTensor(weights)


def discover_videos(data_root: str, dataset_name: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Auto-discover video files from a dataset directory.
    
    Expected structures:
      data/rwf/train/Violence/*.avi
      data/rwf/train/NonViolence/*.avi
      data/rwf/val/Violence/*.avi
      ...
    OR:
      data/rwf/Violence/*.avi
      data/rwf/NonViolence/*.avi
    """
    video_exts = {'.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    dataset_dir = os.path.join(data_root, dataset_name)
    
    paths, labels, sources = [], [], []
    
    if not os.path.isdir(dataset_dir):
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return paths, labels, sources
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in video_exts:
                continue
            
            filepath = os.path.join(root, f)
            # Determine label from directory hierarchy
            parent = os.path.basename(root).lower()
            grandparent = os.path.basename(os.path.dirname(root)).lower()
            combined_tag = f"{grandparent}/{parent}"
            
            label = None
            if "non" in combined_tag or "normal" in combined_tag:
                label = 0
            elif "violence" in combined_tag or "fight" in combined_tag:
                label = 1
            
            if label is not None:
                paths.append(filepath)
                labels.append(label)
                sources.append(dataset_name)
    
    logger.info(f"[{dataset_name}] Found {len(paths)} videos: "
                f"{sum(1 for l in labels if l==0)} non-violence, "
                f"{sum(1 for l in labels if l==1)} violence")
    return paths, labels, sources


def split_dataset(
    paths: List[str],
    labels: List[int],
    sources: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple]:
    """Split dataset at VIDEO level (not frame level) to avoid data leakage."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Group by label for stratified split
    rng = random.Random(seed)
    
    indices_0 = [i for i, l in enumerate(labels) if l == 0]
    indices_1 = [i for i, l in enumerate(labels) if l == 1]
    
    rng.shuffle(indices_0)
    rng.shuffle(indices_1)
    
    def _split_indices(indices):
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (indices[:n_train],
                indices[n_train:n_train + n_val],
                indices[n_train + n_val:])
    
    train_0, val_0, test_0 = _split_indices(indices_0)
    train_1, val_1, test_1 = _split_indices(indices_1)
    
    train_idx = train_0 + train_1
    val_idx = val_0 + val_1
    test_idx = test_0 + test_1
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    def _select(indices):
        return ([paths[i] for i in indices],
                [labels[i] for i in indices],
                [sources[i] for i in indices])
    
    return {
        "train": _select(train_idx),
        "val": _select(val_idx),
        "test": _select(test_idx),
    }


def build_dataloaders(
    config: dict,
    dataset_name: str = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train/val/test DataLoaders from config.
    Returns loaders and dataset info dict.
    """
    cfg_ds = config["dataset"]
    cfg_tr = config["training"]
    cfg_norm = config["normalization"]
    
    ds_name = dataset_name or cfg_ds["name"]
    data_root = cfg_ds["data_root"]
    
    # Discover videos
    if ds_name == "combined":
        all_paths, all_labels, all_sources = [], [], []
        for sub in ["rwf", "avd", "sfd"]:
            p, l, s = discover_videos(data_root, sub)
            all_paths.extend(p)
            all_labels.extend(l)
            all_sources.extend(s)
    else:
        all_paths, all_labels, all_sources = discover_videos(data_root, ds_name)
    
    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"No videos found for dataset '{ds_name}' in {data_root}/{ds_name}/. "
            f"Expected structure: {data_root}/{ds_name}/Violence/*.avi and "
            f"{data_root}/{ds_name}/NonViolence/*.avi"
        )
    
    # Check if pre-split manifest files exist in splits/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(project_root, "splits")
    manifest_train_csv = os.path.join(splits_dir, f"{ds_name}_train.csv")
    manifest_train_txt = os.path.join(splits_dir, f"{ds_name}_train.txt")

    if os.path.isfile(manifest_train_csv) or os.path.isfile(manifest_train_txt):
        logger.info(f"Loading verified split manifests from {splits_dir}/ for {ds_name}...")
        splits = {}
        for sname in ["train", "val", "test"]:
            csv_path = os.path.join(splits_dir, f"{ds_name}_{sname}.csv")
            txt_path = os.path.join(splits_dir, f"{ds_name}_{sname}.txt")
            m_paths, m_labels, m_sources = [], [], []
            if os.path.isfile(csv_path):
                import csv
                with open(csv_path, "r") as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        m_paths.append(row["absolute_path"])
                        m_labels.append(int(row["label"]))
                        m_sources.append(row.get("dataset", ds_name))
            elif os.path.isfile(txt_path):
                with open(txt_path, "r") as mf:
                    m_paths = [line.strip() for line in mf if line.strip()]
                for mp in m_paths:
                    p_lower = mp.lower()
                    m_labels.append(0 if "non" in p_lower else 1)
                    m_sources.append(ds_name)
            splits[sname] = (m_paths, m_labels, m_sources)
    elif has_presplit:
        # Use existing splits
        splits = {}
        for split_name in ["train", "val", "test"]:
            # Try different capitalizations
            for try_name in [split_name, split_name.capitalize()]:
                split_dir = os.path.join(data_root, ds_name, try_name)
                if os.path.isdir(split_dir):
                    p, l, s = discover_videos(data_root, os.path.join(ds_name, try_name))
                    # Fix source back to dataset name
                    s = [ds_name] * len(p)
                    splits[split_name] = (p, l, s)
                    break
            if split_name not in splits:
                splits[split_name] = ([], [], [])
        
        # If no val split, create from train
        if len(splits.get("val", ([], [], []))[0]) == 0 and len(splits["train"][0]) > 0:
            tp, tl, ts = splits["train"]
            n = len(tp)
            n_val = max(1, int(n * 0.15))
            combined = list(zip(tp, tl, ts))
            random.Random(42).shuffle(combined)
            tp, tl, ts = zip(*combined)
            splits["val"] = (list(tp[:n_val]), list(tl[:n_val]), list(ts[:n_val]))
            splits["train"] = (list(tp[n_val:]), list(tl[n_val:]), list(ts[n_val:]))
    else:
        # Create our own splits
        splits = split_dataset(
            all_paths, all_labels, all_sources,
            train_ratio=cfg_ds.get("train_split", 0.7),
            val_ratio=cfg_ds.get("val_split", 0.15),
            test_ratio=cfg_ds.get("test_split", 0.15),
            seed=config["experiment"].get("seed", 42),
        )
    
    mean = tuple(cfg_norm["mean"])
    std = tuple(cfg_norm["std"])
    
    # Create datasets
    train_ds = VideoDataset(
        *splits["train"],
        sequence_length=cfg_ds["sequence_length"],
        image_size=cfg_ds["image_size"],
        sampling_strategy=cfg_ds.get("sampling_strategy", "uniform"),
        is_training=True,
        normalize_mean=mean,
        normalize_std=std,
    )
    
    val_ds = VideoDataset(
        *splits["val"],
        sequence_length=cfg_ds["sequence_length"],
        image_size=cfg_ds["image_size"],
        sampling_strategy="uniform",
        is_training=False,
        normalize_mean=mean,
        normalize_std=std,
    )
    
    test_ds = VideoDataset(
        *splits["test"],
        sequence_length=cfg_ds["sequence_length"],
        image_size=cfg_ds["image_size"],
        sampling_strategy="uniform",
        is_training=False,
        normalize_mean=mean,
        normalize_std=std,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_tr["batch_size"],
        shuffle=True,
        num_workers=cfg_tr.get("num_workers", 4),
        pin_memory=cfg_tr.get("pin_memory", True),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_tr["batch_size"],
        shuffle=False,
        num_workers=cfg_tr.get("num_workers", 4),
        pin_memory=cfg_tr.get("pin_memory", True),
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg_tr["batch_size"],
        shuffle=False,
        num_workers=cfg_tr.get("num_workers", 4),
        pin_memory=cfg_tr.get("pin_memory", True),
    )
    
    info = {
        "dataset": ds_name,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
        "train_violence": sum(1 for l in splits["train"][1] if l == 1),
        "train_nonviolence": sum(1 for l in splits["train"][1] if l == 0),
        "val_violence": sum(1 for l in splits["val"][1] if l == 1),
        "val_nonviolence": sum(1 for l in splits["val"][1] if l == 0),
        "test_violence": sum(1 for l in splits["test"][1] if l == 1),
        "test_nonviolence": sum(1 for l in splits["test"][1] if l == 0),
        "class_weights": train_ds.get_class_weights().tolist(),
    }
    
    return train_loader, val_loader, test_loader, info

