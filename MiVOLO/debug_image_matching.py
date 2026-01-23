#!/usr/bin/env python
"""
Diagnostic script to identify why images are not being found.
This script compares CSV image_numbers with actual filenames in the directory.
"""

import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Debug image matching issues")
    parser.add_argument("--input", type=str, default="/media/ee303/disk2/ffhq_aging_70K/images1024x1024", help="Image directory")
    parser.add_argument("--csv-input", type=str, default='/media/ee303/disk2/ffhq_aging_70K/age_grouping_output.csv', help="CSV file")
    args = parser.parse_args()

    # Read CSV
    print(f"[1] Reading CSV from {args.csv_input}...")
    df = pd.read_csv(args.csv_input, dtype=str)
    print(f"    Total rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Filter age_group == 8
    df_target = df[df['age_group'].astype(str) == '8']
    image_numbers = df_target['image_number'].astype(str).tolist()
    print(f"\n[2] Filtered to age_group==8: {len(image_numbers)} entries")
    
    # Show first few image_number values with repr to catch hidden characters
    print("\n    First 5 image_number values (with repr):")
    for i, num in enumerate(image_numbers[:5]):
        print(f"      [{i}] {repr(num)}")

    # List all files in input directory
    print(f"\n[3] Listing files in {args.input}...")
    try:
        all_files = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        print(f"    Total files: {len(all_files)}")
        print(f"    Image files: {len(image_files)}")
        
        # Extract basenames (without extension)
        basenames = {}
        for img_file in image_files:
            name, ext = os.path.splitext(img_file)
            basenames[name] = img_file
        
        print(f"    Unique basenames: {len(basenames)}")
        print("\n    First 10 basenames:")
        for i, (name, file) in enumerate(list(basenames.items())[:10]):
            print(f"      [{i}] '{name}' -> '{file}'")
    except Exception as e:
        print(f"    ERROR reading directory: {e}")
        return

    # Match image_numbers with basenames
    print(f"\n[4] Attempting to match {len(image_numbers)} image_numbers to basenames...")
    
    found = 0
    not_found = []
    
    for img_num in image_numbers[:50]:  # Test first 50
        img_num_str = str(img_num).strip()
        
        # Strategy 1: exact match
        if img_num_str in basenames:
            found += 1
            continue
        
        # Strategy 2: match after removing leading zeros
        match_found = False
        for name in basenames.keys():
            if name.lstrip('0') == img_num_str.lstrip('0'):
                found += 1
                match_found = True
                print(f"    ✓ '{img_num_str}' matched via lstrip('0'): '{name}'")
                break
        if match_found:
            continue
        
        # Strategy 3: filename contains image_number
        for name in basenames.keys():
            if img_num_str in name:
                found += 1
                match_found = True
                print(f"    ✓ '{img_num_str}' found in basename: '{name}'")
                break
        if match_found:
            continue
        
        # Strategy 4: startswith / endswith
        for name in basenames.keys():
            if name.startswith(img_num_str) or name.endswith(img_num_str):
                found += 1
                match_found = True
                print(f"    ✓ '{img_num_str}' matched via startswith/endswith: '{name}'")
                break
        if match_found:
            continue
        
        # Not found
        not_found.append(img_num_str)
        print(f"    ✗ '{img_num_str}' NOT FOUND (tested with exact, lstrip, contains, startswith/endswith)")
    
    print(f"\n[5] Results (tested first 50):")
    print(f"    Found: {found}/50")
    print(f"    Not found: {len(not_found)}/50")
    
    if not_found:
        print(f"\n    Not found examples: {not_found[:10]}")
        print(f"\n    === Trying fuzzy matching for first not_found ===")
        if not_found:
            sample = not_found[0]
            print(f"    Looking for candidates similar to '{sample}'...")
            candidates = []
            for name in list(basenames.keys())[:100]:  # check first 100
                # Check if numeric part has some overlap
                if any(c in name for c in sample):
                    candidates.append(name)
            print(f"    Candidates with shared chars: {candidates[:20]}")

if __name__ == "__main__":
    main()
