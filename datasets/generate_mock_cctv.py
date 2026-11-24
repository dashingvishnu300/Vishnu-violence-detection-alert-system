"""
Suraksha-Net: Synthetic CCTV Generator for Verification & Pipeline Testing
==========================================================================
Generates a small set of valid synthetic CCTV video clips:
- RWF (Violence & Non-Violence)
- AVD (Violence & Non-Violence with false positive action mock)
- SFD (Surveillance style)
Used to rigorously verify the entire PyTorch DataLoader, CNN-LSTM,
evaluation, and handoff pipeline before or alongside full dataset download.
"""

import os
import cv2
import numpy as np


def create_synthetic_video(file_path: str, is_violence: bool, num_frames: int = 30, width: int = 320, height: int = 240):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, 15.0, (width, height))
    
    # Synthetic CCTV-like scene: static background + moving objects
    bg = np.ones((height, width, 3), dtype=np.uint8) * 120
    # Add some hallway / CCTV lines
    cv2.line(bg, (0, int(height * 0.7)), (width, int(height * 0.7)), (80, 80, 80), 2)
    cv2.line(bg, (int(width * 0.3), 0), (int(width * 0.3), height), (90, 90, 90), 2)
    
    for f in range(num_frames):
        frame = bg.copy()
        # Add timestamp text like real CCTV
        cv2.putText(frame, f"CAM-01 2026-11-21 14:00:{f:02d}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if is_violence:
            # Rapid erratic motion (fighting)
            x1 = int(width * 0.4 + np.random.randint(-30, 30))
            y1 = int(height * 0.5 + np.random.randint(-20, 20))
            x2 = int(width * 0.5 + np.random.randint(-30, 30))
            y2 = int(height * 0.5 + np.random.randint(-20, 20))
            cv2.circle(frame, (x1, y1), 18, (0, 0, 200), -1)
            cv2.circle(frame, (x2, y2), 18, (200, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        else:
            # Smooth gentle motion (walking / standing)
            x1 = int(width * 0.3 + f * 3) % width
            y1 = int(height * 0.5)
            cv2.circle(frame, (x1, y1), 18, (0, 180, 0), -1)
            
        out.write(frame)
        
    out.release()


def generate_pipeline_test_data(data_root: str = "data"):
    print("Generating verification sample videos for pipeline validation...")
    configs = [
        ("rwf/train/Violence", True, 8),
        ("rwf/train/NonViolence", False, 8),
        ("rwf/val/Violence", True, 4),
        ("rwf/val/NonViolence", False, 4),
        ("rwf/test/Violence", True, 4),
        ("rwf/test/NonViolence", False, 4),
        
        ("avd/Violence", True, 4),
        ("avd/NonViolence", False, 4),
        
        ("sfd/Violence", True, 4),
        ("sfd/NonViolence", False, 4),
    ]
    
    for subpath, is_v, count in configs:
        target_dir = os.path.join(data_root, subpath)
        for i in range(count):
            filename = f"sample_{i+1:03d}.mp4"
            create_synthetic_video(os.path.join(target_dir, filename), is_violence=is_v)
            
    print(f"✅ Generated sample video sets in {data_root}/!")


if __name__ == "__main__":
    generate_pipeline_test_data()

