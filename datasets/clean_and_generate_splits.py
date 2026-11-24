"""
Suraksha-Net: Data Sanitizer & Event-Level Split Manifest Generator
===================================================================
1. Purges synthetic sample_*.mp4 clips from data/sfd and data/avd.
2. Implements leak-free event-level splitting for AVD (pairing cam1 and cam2).
3. Implements stratified splitting for SFD.
4. Saves deterministic, reproducible train/val/test video lists in splits/ directory.
"""

import os
import glob
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path("/home/arnav/Documents/vishnu project")
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"


def purge_synthetic_clips():
    """Remove synthetic sample_*.mp4 from sfd and avd."""
    sfd_samples = list(DATA_DIR.glob("sfd/**/sample_*.mp4"))
    avd_samples = list(DATA_DIR.glob("avd/**/sample_*.mp4"))
    
    print(f"Purging {len(sfd_samples)} synthetic clips from SFD...")
    for p in sfd_samples:
        p.unlink()
        
    print(f"Purging {len(avd_samples)} synthetic clips from AVD...")
    for p in avd_samples:
        p.unlink()
        
    print("Purge complete! Real video counts:")
    sfd_real = list(DATA_DIR.glob("sfd/**/*.mp4"))
    avd_real = list(DATA_DIR.glob("avd/**/*.mp4"))
    print(f"  SFD real videos: {len(sfd_real)} (150 Violence, 150 NonViolence expected)")
    print(f"  AVD real videos: {len(avd_real)} (230 Violence, 120 NonViolence expected)")


def generate_sfd_splits(train_r=0.7, val_r=0.15, test_r=0.15, seed=42):
    """Stratified video-level split for SFD."""
    rng = random.Random(seed)
    violence = sorted([str(p) for p in (DATA_DIR / "sfd" / "Violence").glob("*.mp4")])
    nonviolence = sorted([str(p) for p in (DATA_DIR / "sfd" / "NonViolence").glob("*.mp4")])
    
    rng.shuffle(violence)
    rng.shuffle(nonviolence)
    
    def split_list(lst):
        n = len(lst)
        n_tr = int(n * train_r)
        n_val = int(n * val_r)
        return lst[:n_tr], lst[n_tr:n_tr + n_val], lst[n_tr + n_val:]
    
    v_tr, v_val, v_te = split_list(violence)
    nv_tr, nv_val, nv_te = split_list(nonviolence)
    
    return {
        "train": sorted(v_tr + nv_tr),
        "val": sorted(v_val + nv_val),
        "test": sorted(v_te + nv_te),
    }


def generate_avd_splits(train_r=0.7, val_r=0.15, test_r=0.15, seed=42):
    """
    Event-level stratified split for AVD.
    Guarantees that cam1 and cam2 of the same event are assigned to the EXACT same split!
    """
    rng = random.Random(seed)
    all_avd = list((DATA_DIR / "avd").glob("**/*.mp4"))
    
    events: Dict[Tuple[str, str], List[str]] = {}
    for p in all_avd:
        base = p.name
        # Format: avd_{violent|nonviolent}_{cam1|cam2}_{event_id}.mp4
        m = re.search(r"avd_(violent|nonviolent)_(cam\d+)_(\d+)\.mp4", base)
        if m:
            label, cam, event_id = m.groups()
            key = (label, event_id)
            events.setdefault(key, []).append(str(p))
            
    v_events = [ev for ev in events.keys() if ev[0] == "violent"]
    nv_events = [ev for ev in events.keys() if ev[0] == "nonviolent"]
    
    rng.shuffle(v_events)
    rng.shuffle(nv_events)
    
    def split_events(ev_list):
        n = len(ev_list)
        n_tr = int(n * train_r)
        n_val = int(n * val_r)
        return ev_list[:n_tr], ev_list[n_tr:n_tr + n_val], ev_list[n_tr + n_val:]
        
    v_tr, v_val, v_te = split_events(v_events)
    nv_tr, nv_val, nv_te = split_events(nv_events)
    
    def expand_events(ev_list):
        videos = []
        for ev in ev_list:
            videos.extend(events[ev])
        return sorted(videos)
        
    return {
        "train": expand_events(v_tr + nv_tr),
        "val": expand_events(v_val + nv_val),
        "test": expand_events(v_te + nv_te),
    }


def save_manifests(dataset_name: str, splits: Dict[str, List[str]]):
    SPLITS_DIR.mkdir(exist_ok=True)
    for split_type in ["train", "val", "test"]:
        manifest_file = SPLITS_DIR / f"{dataset_name}_{split_type}.txt"
        with open(manifest_file, "w") as f:
            for item in splits[split_type]:
                f.write(f"{item}\n")
        print(f"Saved {len(splits[split_type])} videos to {manifest_file.relative_to(PROJECT_ROOT)}")


def main():
    purge_synthetic_clips()
    
    print("\n--- Generating Clean SFD Splits ---")
    sfd_splits = generate_sfd_splits()
    save_manifests("sfd", sfd_splits)
    
    print("\n--- Generating Event-Grouped AVD Splits (cam1 & cam2 paired) ---")
    avd_splits = generate_avd_splits()
    save_manifests("avd", avd_splits)
    
    print("\n--- Generating Combined (SFD + AVD) Splits ---")
    combined_splits = {
        "train": sfd_splits["train"] + avd_splits["train"],
        "val": sfd_splits["val"] + avd_splits["val"],
        "test": sfd_splits["test"] + avd_splits["test"],
    }
    save_manifests("combined", combined_splits)
    print("All split manifests generated successfully!")


if __name__ == "__main__":
    main()

