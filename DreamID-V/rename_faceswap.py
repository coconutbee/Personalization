#!/usr/bin/env python3
"""
Rename face swap output images
From: {prompt}_{id}.jpg
To: {id}_{prompt}.jpg
"""

import os
import re
from pathlib import Path

def rename_files(directory):
    """Rename files to put ID first"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(directory.glob(ext))
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    print(f"Found {len(image_files)} images")
    renamed_count = 0
    skipped_count = 0
    
    for image_path in sorted(image_files):
        filename = image_path.name
        
        # Match pattern: {prompt}_{id}.ext
        # Look for underscore followed by digits at the end of the filename (before extension)
        match = re.match(r'^(.+)_(\d+)(\.[^.]+)$', filename)
        
        if match:
            prompt_part = match.group(1)
            id_part = match.group(2)
            ext = match.group(3)
            
            # New filename: {id}_{prompt}.ext
            new_filename = f"{id_part}_{prompt_part}{ext}"
            new_path = directory / new_filename
            
            # Check if already renamed or if new name exists
            if new_path.exists():
                print(f"Skip (target exists): {filename}")
                skipped_count += 1
                continue
            
            # Rename
            try:
                image_path.rename(new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
        else:
            print(f"Skip (no match): {filename}")
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Renamed: {renamed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(image_files)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rename_faceswap.py <directory>")
        print("Example: python rename_faceswap.py FACESWAP_PIXART")
        sys.exit(1)
    
    directory = sys.argv[1]
    rename_files(directory)
