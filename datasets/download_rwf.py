"""
Suraksha-Net: RWF-2000 Downloader & Extractor (Zenodo Record 15687512)
=====================================================================
Downloads the official 13-part 7z archive of RWF-2000:
- Handles resuming partial downloads
- Concatenates and extracts directly into data/rwf/
- Expected result: 2,000 video clips (1,000 Violence, 1,000 Non-Violence)
"""

import os
import sys
import urllib.request
import subprocess
from tqdm import tqdm

ZENODO_BASE = "https://zenodo.org/api/records/15687512/files"
PARTS = [f"{i:03d}" for i in range(1, 14)]


def download_file_with_resume(url: str, output_path: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    file_size = 0
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"Error fetching header for {url}: {e}")
        return False
        
    if file_size >= total_size and total_size > 0:
        print(f"  [Already Downloaded] {os.path.basename(output_path)} ({total_size/1024/1024:.1f} MB)")
        return True
        
    if file_size > 0:
        headers["Range"] = f"bytes={file_size}-"
        req = urllib.request.Request(url, headers=headers)
        
    print(f"  Downloading {os.path.basename(output_path)} ({total_size/1024/1024:.1f} MB)...")
    mode = "ab" if file_size > 0 else "wb"
    
    with urllib.request.urlopen(req) as resp, open(output_path, mode) as out:
        with tqdm(total=total_size, initial=file_size, unit="B", unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out.write(chunk)
                pbar.update(len(chunk))
    return True


def download_and_extract_rwf(download_dir: str = "data/downloads/rwf", target_dir: str = "data/rwf"):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DOWNLOADING RWF-2000 (ZENODO 15687512)")
    print(f"Total parts: 13 | Destination: {download_dir}")
    print("="*60 + "\n")
    
    for p in PARTS:
        filename = f"RWF-2000.7z.{p}"
        url = f"{ZENODO_BASE}/{filename}/content"
        out_path = os.path.join(download_dir, filename)
        success = download_file_with_resume(url, out_path)
        if not success:
            print(f"Failed to download {filename}")
            return
            
    print("\nAll 13 parts downloaded successfully! Extracting archive...")
    first_part = os.path.join(download_dir, "RWF-2000.7z.001")
    
    # Try 7z / py7zr
    try:
        import py7zr
        print("Using py7zr to extract split multi-volume 7z...")
        # Concatenate volumes if needed or extract with py7zr
        with py7zr.SevenZipFile(first_part, mode='r') as archive:
            archive.extractall(path=target_dir)
        print(f"✅ Extracted RWF-2000 to {target_dir}/")
    except Exception as e:
        print(f"py7zr extraction notice: {e}")
        print("Trying 7z CLI or fallback...")
        res = subprocess.run(["7z", "x", first_part, f"-o{target_dir}", "-y"])
        if res.returncode == 0:
            print(f"✅ 7z Extracted RWF-2000 to {target_dir}/")
        else:
            print("Please ensure 7z is installed: sudo apt install p7zip-full")


if __name__ == "__main__":
    download_and_extract_rwf()

