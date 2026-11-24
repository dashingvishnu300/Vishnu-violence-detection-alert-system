"""
Suraksha-Net: Scene-Level Split Generator & Data Manifest Builder
=================================================================
Requirements:
1. Split by scene, not by clip.
2. SFD: Group all clips originating from the same YouTube source or dataset into the same scene.
3. AIRTLab (AVD): Group both camera views (cam1 & cam2) of the same event into the same scene.
4. RWF-2000: Keep official val folder (400 videos) as separate untouched test set.
   Split official train folder (1,576 videos) into train (~85%) and validation (~15%).
5. Save CSV clip lists alongside .txt manifests.
"""

import os
import re
import csv
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path("/home/arnav/Documents/vishnu project").resolve()
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"


def get_sfd_scene_mapping() -> Dict[str, str]:
    """Parse videos.txt to map every SFD clip to its source scene/URL."""
    videos_txt = DATA_DIR / "downloads" / "fight-detection-surv-dataset" / "videos.txt"
    source_map = {}
    current_source = "unknown"
    
    with open(videos_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("http"):
                current_source = line
            elif ":" in line:
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip()
                if " - " in left:
                    m = re.match(r"([a-zA-Z]+)(\d+)\s*-\s*([a-zA-Z]+)(\d+)", left)
                    if m:
                        pfx1, start, pfx2, end = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
                        for idx in range(start, end + 1):
                            source_map[f"{pfx1}{idx:03d}"] = right
                else:
                    source_map[left] = current_source
    return source_map


def build_sfd_splits(seed: int = 42) -> Dict[str, List[dict]]:
    source_map = get_sfd_scene_mapping()
    
    v_files = sorted(list((DATA_DIR / "sfd" / "Violence").glob("*.mp4")))
    nv_files = sorted(list((DATA_DIR / "sfd" / "NonViolence").glob("*.mp4")))
    
    # Group by scene
    scenes: Dict[str, List[dict]] = {}
    for p in v_files:
        name = p.stem
        scene = source_map.get(name, f"sfd_unk_{name}")
        scenes.setdefault(scene, []).append({
            "path": str(p),
            "label": 1,
            "label_name": "Violence",
            "dataset": "sfd",
            "scene_id": f"sfd_{scene}",
        })
        
    for p in nv_files:
        name = p.stem
        scene = source_map.get(name, f"sfd_unk_{name}")
        scenes.setdefault(scene, []).append({
            "path": str(p),
            "label": 0,
            "label_name": "NonViolence",
            "dataset": "sfd",
            "scene_id": f"sfd_{scene}",
        })
        
    # Categorize scenes by majority label
    v_scenes = [sc for sc, items in scenes.items() if any(it["label"] == 1 for it in items)]
    nv_scenes = [sc for sc, items in scenes.items() if all(it["label"] == 0 for it in items)]
    
    rng = random.Random(seed)
    rng.shuffle(v_scenes)
    rng.shuffle(nv_scenes)
    
    def partition(scene_list, r_tr=0.70, r_val=0.15):
        n = len(scene_list)
        n_tr = max(1, int(n * r_tr))
        n_val = max(1, int(n * r_val))
        return scene_list[:n_tr], scene_list[n_tr:n_tr + n_val], scene_list[n_tr + n_val:]
        
    v_tr, v_val, v_te = partition(v_scenes)
    nv_tr, nv_val, nv_te = partition(nv_scenes)
    
    def collect(sc_list):
        out = []
        for sc in sc_list:
            out.extend(scenes[sc])
        return sorted(out, key=lambda x: x["path"])
        
    return {
        "train": collect(v_tr + nv_tr),
        "val": collect(v_val + nv_val),
        "test": collect(v_te + nv_te),
    }


def build_avd_splits(seed: int = 42) -> Dict[str, List[dict]]:
    all_avd = sorted(list((DATA_DIR / "avd").glob("**/*.mp4")))
    
    events: Dict[Tuple[str, str], List[dict]] = {}
    for p in all_avd:
        m = re.search(r"avd_(violent|nonviolent)_(cam\d+)_(\d+)\.mp4", p.name)
        if m:
            lbl_str, cam, event_id = m.groups()
            is_v = 1 if lbl_str == "violent" else 0
            lbl_name = "Violence" if is_v == 1 else "NonViolence"
            key = (lbl_str, event_id)
            events.setdefault(key, []).append({
                "path": str(p),
                "label": is_v,
                "label_name": lbl_name,
                "dataset": "avd",
                "scene_id": f"avd_event_{lbl_str}_{event_id}",
            })
            
    v_events = [ev for ev in events.keys() if ev[0] == "violent"]
    nv_events = [ev for ev in events.keys() if ev[0] == "nonviolent"]
    
    rng = random.Random(seed)
    rng.shuffle(v_events)
    rng.shuffle(nv_events)
    
    def partition(ev_list, r_tr=0.70, r_val=0.15):
        n = len(ev_list)
        n_tr = int(n * r_tr)
        n_val = int(n * r_val)
        return ev_list[:n_tr], ev_list[n_tr:n_tr + n_val], ev_list[n_tr + n_val:]
        
    v_tr, v_val, v_te = partition(v_events)
    nv_tr, nv_val, nv_te = partition(nv_events)
    
    def collect(ev_list):
        out = []
        for ev in ev_list:
            out.extend(events[ev])
        return sorted(out, key=lambda x: x["path"])
        
    return {
        "train": collect(v_tr + nv_tr),
        "val": collect(v_val + nv_val),
        "test": collect(v_te + nv_te),
    }


def build_rwf_splits(seed: int = 42) -> Dict[str, List[dict]]:
    # Official test set: val folder (400 videos)
    val_fight = sorted(list((DATA_DIR / "rwf" / "val" / "Violence").glob("*.*")))
    val_nonfight = sorted(list((DATA_DIR / "rwf" / "val" / "NonViolence").glob("*.*")))
    
    test_clips = []
    for p in val_fight:
        stem = p.stem
        vid_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
        test_clips.append({
            "path": str(p),
            "label": 1,
            "label_name": "Violence",
            "dataset": "rwf",
            "scene_id": f"rwf_scene_{vid_id}",
        })
    for p in val_nonfight:
        stem = p.stem
        vid_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
        test_clips.append({
            "path": str(p),
            "label": 0,
            "label_name": "NonViolence",
            "dataset": "rwf",
            "scene_id": f"rwf_scene_{vid_id}",
        })
        
    # Official train folder (1576 videos): group by video source ID (scene)
    tr_fight = sorted(list((DATA_DIR / "rwf" / "train" / "Violence").glob("*.*")))
    tr_nonfight = sorted(list((DATA_DIR / "rwf" / "train" / "NonViolence").glob("*.*")))
    
    scenes: Dict[str, List[dict]] = {}
    for p in tr_fight:
        stem = p.stem
        vid_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
        scenes.setdefault(vid_id, []).append({
            "path": str(p),
            "label": 1,
            "label_name": "Violence",
            "dataset": "rwf",
            "scene_id": f"rwf_scene_{vid_id}",
        })
    for p in tr_nonfight:
        stem = p.stem
        vid_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
        scenes.setdefault(vid_id, []).append({
            "path": str(p),
            "label": 0,
            "label_name": "NonViolence",
            "dataset": "rwf",
            "scene_id": f"rwf_scene_{vid_id}",
        })
        
    # Stratified scene split (~85% train, ~15% val)
    scene_keys = list(scenes.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_keys)
    
    # Balance scenes based on whether they contain violence
    v_scenes = [sc for sc in scene_keys if any(it["label"] == 1 for it in scenes[sc])]
    nv_scenes = [sc for sc in scene_keys if all(it["label"] == 0 for it in scenes[sc])]
    
    def split_scenes(sc_list, train_ratio=0.85):
        n = len(sc_list)
        n_tr = int(n * train_ratio)
        return sc_list[:n_tr], sc_list[n_tr:]
        
    v_tr, v_val = split_scenes(v_scenes)
    nv_tr, nv_val = split_scenes(nv_scenes)
    
    train_clips = []
    for sc in v_tr + nv_tr:
        train_clips.extend(scenes[sc])
        
    val_clips = []
    for sc in v_val + nv_val:
        val_clips.extend(scenes[sc])
        
    return {
        "train": sorted(train_clips, key=lambda x: x["path"]),
        "val": sorted(val_clips, key=lambda x: x["path"]),
        "test": sorted(test_clips, key=lambda x: x["path"]),
    }


def save_dataset_manifests(name: str, splits: Dict[str, List[dict]]):
    SPLITS_DIR.mkdir(exist_ok=True)
    fieldnames = ["relative_path", "absolute_path", "label", "label_name", "dataset", "scene_id"]
    
    for split_type in ["train", "val", "test"]:
        items = splits[split_type]
        csv_path = SPLITS_DIR / f"{name}_{split_type}.csv"
        txt_path = SPLITS_DIR / f"{name}_{split_type}.txt"
        
        with open(csv_path, "w", newline="") as f_csv, open(txt_path, "w") as f_txt:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for it in items:
                rel = os.path.relpath(it["path"], PROJECT_ROOT)
                row = {
                    "relative_path": rel,
                    "absolute_path": it["path"],
                    "label": it["label"],
                    "label_name": it["label_name"],
                    "dataset": it["dataset"],
                    "scene_id": it["scene_id"],
                }
                writer.writerow(row)
                f_txt.write(f"{it['path']}\n")
                
        print(f"[{name}] {split_type.upper():5s}: {len(items):4d} clips -> {csv_path.name}")


def main():
    print("=== Generating Scene-Level Leak-Free Splits ===")
    sfd_splits = build_sfd_splits()
    save_dataset_manifests("sfd", sfd_splits)
    
    print("\n=== Generating Event-Grouped AVD Splits ===")
    avd_splits = build_avd_splits()
    save_dataset_manifests("avd", avd_splits)
    
    print("\n=== Generating RWF-2000 Splits (Official Val as Test Set) ===")
    rwf_splits = build_rwf_splits()
    save_dataset_manifests("rwf", rwf_splits)
    
    print("\n=== Generating Combined 3-Dataset Splits (SFD + AVD + RWF) ===")
    combined_splits = {
        "train": sfd_splits["train"] + avd_splits["train"] + rwf_splits["train"],
        "val": sfd_splits["val"] + avd_splits["val"] + rwf_splits["val"],
        "test": sfd_splits["test"] + avd_splits["test"] + rwf_splits["test"],
    }
    save_dataset_manifests("combined", combined_splits)
    
    # Also save standard top-level CSV manifests: train_clips.csv, val_clips.csv, test_clips.csv
    for split_type in ["train", "val", "test"]:
        src = SPLITS_DIR / f"combined_{split_type}.csv"
        dst = SPLITS_DIR / f"{split_type}_clips.csv"
        dst.write_text(src.read_text())
        print(f"Created global {dst.name}")
        
    print("\n✅ All scene-level manifests and CSVs successfully generated!")


if __name__ == "__main__":
    main()
