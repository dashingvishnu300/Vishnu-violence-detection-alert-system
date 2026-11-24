"""
Suraksha-Net: Dataset Organizer & Ingestion Pipeline
===================================================
Automatically links or copies the official cloned datasets into standard Suraksha-Net structure:
- data/sfd/Violence/ and data/sfd/NonViolence/
- data/avd/Violence/ and data/avd/NonViolence/
- data/rwf/train/ and data/rwf/val/ (once extracted)
"""

import os
import shutil
import glob
from pathlib import Path


def organize_sfd(downloads_dir: str = "data/downloads", target_dir: str = "data/sfd"):
    src = os.path.join(downloads_dir, "fight-detection-surv-dataset")
    if not os.path.isdir(src):
        print(f"SFD source not found in {src}")
        return
        
    os.makedirs(os.path.join(target_dir, "Violence"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "NonViolence"), exist_ok=True)
    
    # 1. Fight -> Violence
    fight_vids = glob.glob(os.path.join(src, "fight", "*.mp4"))
    for f in fight_vids:
        dst = os.path.join(target_dir, "Violence", os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
            
    # 2. noFight -> NonViolence
    nofight_vids = glob.glob(os.path.join(src, "noFight", "*.mp4"))
    for f in nofight_vids:
        dst = os.path.join(target_dir, "NonViolence", os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
            
    print(f"✅ SFD / SCFD Organized: {len(fight_vids)} Violence, {len(nofight_vids)} NonViolence in {target_dir}/")


def organize_avd(downloads_dir: str = "data/downloads", target_dir: str = "data/avd"):
    src = os.path.join(downloads_dir, "A-Dataset-for-Automatic-Violence-Detection-in-Videos", "violence-detection-dataset")
    if not os.path.isdir(src):
        print(f"AVD source not found in {src}")
        return
        
    os.makedirs(os.path.join(target_dir, "Violence"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "NonViolence"), exist_ok=True)
    
    # Violent: cam1 and cam2
    violent_vids = glob.glob(os.path.join(src, "violent", "*", "*.mp4"))
    for f in violent_vids:
        cam = os.path.basename(os.path.dirname(f))
        name = f"avd_violent_{cam}_{os.path.basename(f)}"
        dst = os.path.join(target_dir, "Violence", name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
            
    # Non-violent: cam1 and cam2
    nonviolent_vids = glob.glob(os.path.join(src, "non-violent", "*", "*.mp4"))
    for f in nonviolent_vids:
        cam = os.path.basename(os.path.dirname(f))
        name = f"avd_nonviolent_{cam}_{os.path.basename(f)}"
        dst = os.path.join(target_dir, "NonViolence", name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
            
    print(f"✅ AVD / AIRTLab Organized: {len(violent_vids)} Violence, {len(nonviolent_vids)} NonViolence in {target_dir}/")


def main():
    organize_sfd()
    organize_avd()


if __name__ == "__main__":
    main()

