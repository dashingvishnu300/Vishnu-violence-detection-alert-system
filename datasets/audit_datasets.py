"""
Suraksha-Net: Dataset Auditor & Verifier
=======================================
Performs comprehensive dataset health check:
- Total video count per dataset
- Class distribution (Violence vs. Non-Violence)
- Resolution, duration, FPS statistics
- Detects missing or corrupted video files
- Generates dataset_audit_report.json
"""

import os
import sys
import glob
import json
import cv2
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def audit_directory(data_dir: str):
    print(f"\n========================================================")
    print(f"AUDITING DATASETS IN: {data_dir}")
    print(f"========================================================\n")
    
    datasets = ["rwf", "avd", "sfd"]
    stats = {}
    video_exts = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
    
    for ds in datasets:
        ds_path = os.path.join(data_dir, ds)
        if not os.path.exists(ds_path):
            stats[ds] = {"status": "MISSING", "path": ds_path}
            continue
            
        all_files = []
        for root, _, files in os.walk(ds_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in video_exts:
                    all_files.append(os.path.join(root, f))
                    
        if len(all_files) == 0:
            stats[ds] = {"status": "EMPTY", "path": ds_path, "file_count": 0}
            continue
            
        violence_count = 0
        nonviolence_count = 0
        corrupt_files = []
        resolutions = defaultdict(int)
        durations = []
        fps_list = []
        
        print(f"Analyzing {len(all_files)} videos in {ds.upper()}...")
        for vf in tqdm(all_files, desc=f"Scanning {ds}"):
            parent = os.path.basename(os.path.dirname(vf)).lower()
            if "non" in parent or "normal" in parent:
                nonviolence_count += 1
            elif "violence" in parent or "fight" in parent:
                violence_count += 1
            else:
                # check grand-parent
                gp = os.path.basename(os.path.dirname(os.path.dirname(vf))).lower()
                if "non" in gp:
                    nonviolence_count += 1
                elif "violence" in gp or "fight" in gp:
                    violence_count += 1
                else:
                    nonviolence_count += 1  # fallback
                    
            cap = cv2.VideoCapture(vf)
            if not cap.isOpened():
                corrupt_files.append(vf)
                continue
                
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (frame_count / fps) if fps and fps > 0 else 0
            
            resolutions[f"{w}x{h}"] += 1
            if duration > 0:
                durations.append(duration)
            if fps > 0:
                fps_list.append(fps)
            cap.release()
            
        stats[ds] = {
            "status": "READY",
            "total_videos": len(all_files),
            "violence_videos": violence_count,
            "non_violence_videos": nonviolence_count,
            "corrupt_files_count": len(corrupt_files),
            "corrupt_files": corrupt_files,
            "resolutions": dict(resolutions),
            "avg_fps": round(float(sum(fps_list) / len(fps_list)), 2) if fps_list else 0.0,
            "avg_duration_sec": round(float(sum(durations) / len(durations)), 2) if durations else 0.0
        }
        
    report_path = os.path.join(data_dir, "dataset_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"\n========================================================")
    print("DATASET AUDIT SUMMARY:")
    for ds, d in stats.items():
        print(f"\n[{ds.upper()}]: {d['status']}")
        if d['status'] == "READY":
            print(f"  Total Videos: {d['total_videos']}")
            print(f"  Class Balance: {d['violence_videos']} Violence vs {d['non_violence_videos']} Non-Violence")
            print(f"  Corrupt Videos: {d['corrupt_files_count']}")
            print(f"  Avg Duration: {d['avg_duration_sec']}s | Avg FPS: {d['avg_fps']}")
            print(f"  Primary Resolutions: {list(d['resolutions'].keys())[:3]}")
    print(f"\nFull audit written to: {report_path}")
    print(f"========================================================\n")


if __name__ == "__main__":
    audit_directory("data")

