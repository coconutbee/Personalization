import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoImageProcessor, AutoModel
)
import torch.nn.functional as F

# ==========================================
# 0. Environment Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Model Loading (Executed Once)
# ==========================================
def load_models():
    print("\n" + "="*50)
    print("🚀 Initializing Models...")
    print("="*50)
    
    print("🔹 Loading CLIP (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("🔹 Loading DINOv2 (facebook/dinov2-base)...")
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    
    return clip_model, clip_proc, dino_model, dino_proc

# ==========================================
# 2. Metric Calculation Functions
# ==========================================
def get_clip_t2i_score(model, processor, image, text):
    """CLIP Text-to-Image Alignment"""
    try:
        inputs = processor(
            text=[text], images=image, return_tensors="pt", 
            padding=True, truncation=True, max_length=77
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        score = (image_embeds @ text_embeds.t()).item()
        return max(0.0, score)
    except Exception as e:
        return None

def get_clip_i2i_score(model, processor, img1, img2):
    """CLIP Image-to-Image Similarity"""
    try:
        inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        embeds = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        score = (embeds[0] @ embeds[1]).item()
        return max(0.0, score)
    except Exception as e:
        return None

def get_dino_score(model, processor, img1, img2):
    """DINOv2 Image-to-Image Similarity"""
    try:
        inputs1 = processor(images=img1, return_tensors="pt").to(DEVICE)
        inputs2 = processor(images=img2, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out1 = model(**inputs1).last_hidden_state[:, 0, :]
            out2 = model(**inputs2).last_hidden_state[:, 0, :]
        score = F.cosine_similarity(out1, out2).item()
        return max(0.0, score)
    except Exception as e:
        return None

# ==========================================
# 3. File Search Utilities
# ==========================================
def smart_find_image(base_dir, filename):
    if not filename: return None
    name_no_ext = os.path.splitext(filename)[0]
    candidates = [
        filename, f"0_{filename}", 
        f"0_{name_no_ext}.png", f"{name_no_ext}.png", 
        f"{name_no_ext}.jpg"
    ]
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path, cand # Return path and found filename
    return None, None

def find_target_by_prompt(base_dir, prompt):
    if not prompt or not os.path.exists(base_dir): return None
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if normalize(os.path.splitext(filename)[0]) == target_norm:
            return os.path.join(base_dir, filename)
    return None

def smart_find_ref_image(ref_dir, ref_index_str):
    candidates = [
        f"{ref_index_str}.jpg", f"{ref_index_str}.png", f"{ref_index_str}.jpeg",
        f"0_{ref_index_str}.jpg"
    ]
    for cand in candidates:
        full_path = os.path.join(ref_dir, cand)
        if os.path.exists(full_path):
            return full_path
    return None

# ==========================================
# 4. Core Task Processing
# ==========================================
def process_task(task_name, swapped_dir, t2i_dir, ref_dir, json_path, models):
    clip_model, clip_proc, dino_model, dino_proc = models
    
    print(f"\n🔹 Processing Task: [{task_name}]")
    print(f"   📂 Swapped Dir: {swapped_dir}")
    print(f"   📂 T2I Source:  {t2i_dir}")
    print(f"   📂 Reference:   {ref_dir}")

    if not os.path.exists(json_path):
        print(f"   ❌ JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # Statistics Containers
    stats = {
        'clip_t2i_orig': [], 'clip_t2i_swap': [],   # Prompt Alignment
        'clip_struct': [], 'dino_struct': [],       # Structure (Orig vs Swap)
        'clip_id_t2i': [], 'clip_id_swap': [],      # ID Similarity (Ref vs T2I/Swap)
        'dino_id_t2i': [], 'dino_id_swap': []       # ID Similarity (Ref vs T2I/Swap)
    }

    for item in tqdm(data_list, desc=f"   Running {task_name}"):
        raw_filename = item.get('image', '').strip()
        prompt = item.get('prompt', '').strip()

        # Initialize JSON fields
        item['clip_t2i_orig'] = None
        item['clip_t2i_swap'] = None
        item['clip_struct_score'] = None
        item['dino_struct_score'] = None
        item['clip_id_t2i'] = None
        item['clip_id_swap'] = None
        item['dino_id_t2i'] = None
        item['dino_id_swap'] = None

        # --- 1. Load Images ---
        # A. Swapped Image
        swapped_path, found_name = smart_find_image(swapped_dir, raw_filename)
        img_swapped = None
        if swapped_path:
            try: img_swapped = Image.open(swapped_path).convert("RGB")
            except: pass

        # B. T2I Original Image
        t2i_path = find_target_by_prompt(t2i_dir, prompt)
        img_orig = None
        if t2i_path:
            try: img_orig = Image.open(t2i_path).convert("RGB")
            except: pass
            
        # C. Reference Image
        ref_idx_str = "0"
        if found_name:
            try:
                fname_no_ext = os.path.splitext(found_name)[0]
                parts = fname_no_ext.split('_')
                ref_idx_str = parts[0] if len(parts) >= 2 else parts[0]
            except: pass
        
        ref_path = smart_find_ref_image(ref_dir, ref_idx_str)
        img_ref = None
        if ref_path:
            try: img_ref = Image.open(ref_path).convert("RGB")
            except: pass

        # --- 2. Calculate Metrics ---
        
        # [Metric 1] Prompt Alignment (T2I)
        if prompt:
            if img_orig:
                s = get_clip_t2i_score(clip_model, clip_proc, img_orig, prompt)
                item['clip_t2i_orig'] = float(f'{s:.4f}')
                if s: stats['clip_t2i_orig'].append(s)
            
            if img_swapped:
                s = get_clip_t2i_score(clip_model, clip_proc, img_swapped, prompt)
                item['clip_t2i_swap'] = float(f'{s:.4f}')
                if s: stats['clip_t2i_swap'].append(s)

        # [Metric 2] Structure Preservation (Orig vs Swapped)
        if img_orig and img_swapped:
            # CLIP Structure
            sc = get_clip_i2i_score(clip_model, clip_proc, img_orig, img_swapped)
            item['clip_struct_score'] = float(f'{sc:.4f}')
            if sc: stats['clip_struct'].append(sc)
            
            # DINO Structure
            sd = get_dino_score(dino_model, dino_proc, img_orig, img_swapped)
            item['dino_struct_score'] = float(f'{sd:.4f}')
            if sd: stats['dino_struct'].append(sd)

        # [Metric 3] Identity Similarity (Ref vs T2I / Ref vs Swapped) [NEW]
        if img_ref:
            # Baseline ID (Ref vs T2I)
            if img_orig:
                # CLIP ID
                cid_t = get_clip_i2i_score(clip_model, clip_proc, img_ref, img_orig)
                item['clip_id_t2i'] = float(f'{cid_t:.4f}')
                if cid_t: stats['clip_id_t2i'].append(cid_t)
                
                # DINO ID
                did_t = get_dino_score(dino_model, dino_proc, img_ref, img_orig)
                item['dino_id_t2i'] = float(f'{did_t:.4f}')
                if did_t: stats['dino_id_t2i'].append(did_t)

            # Swapped ID (Ref vs Swapped)
            if img_swapped:
                # CLIP ID
                cid_s = get_clip_i2i_score(clip_model, clip_proc, img_ref, img_swapped)
                item['clip_id_swap'] = float(f'{cid_s:.4f}')
                if cid_s: stats['clip_id_swap'].append(cid_s)
                
                # DINO ID
                did_s = get_dino_score(dino_model, dino_proc, img_ref, img_swapped)
                item['dino_id_swap'] = float(f'{did_s:.4f}')
                if did_s: stats['dino_id_swap'].append(did_s)

    # Save Results
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    
    return {
        'name': task_name,
        't2i_orig': avg(stats['clip_t2i_orig']),
        't2i_swap': avg(stats['clip_t2i_swap']),
        'struct_clip': avg(stats['clip_struct']),
        'struct_dino': avg(stats['dino_struct']),
        'clip_id_t2i': avg(stats['clip_id_t2i']),
        'clip_id_swap': avg(stats['clip_id_swap']),
        'dino_id_t2i': avg(stats['dino_id_t2i']),
        'dino_id_swap': avg(stats['dino_id_swap'])
    }

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    # 1. Load Models
    models = load_models()

    # 2. Configuration
    DEFAULT_T2I_DIR = './pixart_outputs'
    REFERENCE_DIR = './faceswap_results/reference' # Need reference images now
    SOURCE_JSON = 'gt.json'
    
    METHOD_DIRS = {
        'PixArt': './faceswap_results/pixart',
        # 'Janus': './faceswap_results/janus',
        # 'Infinity': './faceswap_results/infinity',
        # 'ShowO2': './faceswap_results/showo2'
    }

    results_summary = []
    print(f"\n📋 Starting Batch Processing (Source: {SOURCE_JSON})...")

    for name, swapped_dir in METHOD_DIRS.items():
        if not os.path.exists(swapped_dir):
            print(f"⚠️ Skipping {name}: Directory not found")
            continue

        res = process_task(name, swapped_dir, DEFAULT_T2I_DIR, REFERENCE_DIR, SOURCE_JSON, models)
        if res: results_summary.append(res)

    # 3. Final Summary Table
    print("\n" + "="*145)
    print(f"{'Method':<10} | {'T2I(Orig)':<9} {'T2I(Swap)':<9} | {'Struct(C)':<9} {'Struct(D)':<9} | {'C-ID(Orig)':<10} {'C-ID(Swap)':<10} | {'D-ID(Orig)':<10} {'D-ID(Swap)':<10}")
    print("-" * 145)
    for res in results_summary:
        print(f"{res['name']:<10} | "
              f"{res['t2i_orig']:<9.4f} {res['t2i_swap']:<9.4f} | "
              f"{res['struct_clip']:<9.4f} {res['struct_dino']:<9.4f} | "
              f"{res['clip_id_t2i']:<10.4f} {res['clip_id_swap']:<10.4f} | "
              f"{res['dino_id_t2i']:<10.4f} {res['dino_id_swap']:<10.4f}")
    print("="*145)
    print("✅ All tasks completed.")