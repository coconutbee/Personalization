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
# 3. File Search Utilities (Modified)
# ==========================================
def smart_find_swapped_image(base_dir, prompt, target_id):
    """
    根據 prompt 和 ID 找尋 swapped 資料夾中的圖片。
    格式預設為: "{target_id}_{Prompt}.jpg"
    """
    if not os.path.exists(base_dir): return None, None
    
    # 預處理目標 ID (補零)
    target_id_str = str(target_id).zfill(5)
    
    for filename in os.listdir(base_dir):
        if "_" in filename:
            parts = filename.split('_', 1) 
            file_id = parts[0]
            name_part = os.path.splitext(parts[1])[0]
            
            # 1. 檢查 ID 是否匹配 (這是關鍵修正!)
            if file_id != target_id_str:
                continue

            # 2. 檢查 Prompt 是否匹配
            if name_part.strip() == prompt.strip():
                return os.path.join(base_dir, filename), file_id
                
    return None, None

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 找尋對應的 T2I 原圖"""
    if not prompt or not os.path.exists(base_dir): return None
    
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')): continue
        
        name_no_ext = os.path.splitext(filename)[0]
        # 有些 T2I 圖可能有 "0_" 前綴，這裡做個容錯
        clean_name = name_no_ext[2:] if name_no_ext.startswith("0_") else name_no_ext
        
        if normalize(clean_name) == target_norm:
            return os.path.join(base_dir, filename)
    return None

def smart_find_ref_image(ref_dir, ref_index_str):
    """根據 ID 找尋 Reference 圖片"""
    if not ref_dir or not ref_index_str: return None
    candidates = [
        f"{ref_index_str}.jpg", f"{ref_index_str}.png", f"{ref_index_str}.jpeg",
        f"0_{ref_index_str}.jpg", f"0_{ref_index_str}.png" # 增加 PNG 容錯
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
        't2i_clip_t2i': [], 'swap_clip_t2i': [],   # Prompt Alignment
        'clip_i2i': [], 'dino_i2i': [],       # Structure (Orig vs Swap)
        't2i_clip_id_i2i': [], 'swap_clip_id_i2i': [],      # ID Similarity (Ref vs T2I/Swap)
        't2i_dino_id_i2i': [], 'swap_dino_id_i2i': []       # ID Similarity (Ref vs T2I/Swap)
    }
    
    # 紀錄找不到的圖
    missing_swaps = []

    for item in tqdm(data_list, desc=f"   Running {task_name}"):
        prompt = item.get('prompt', '').strip()
        # 取得 JSON 中的 ID，預設為 '0'
        target_id = item.get('rand_id', '0')

        # Initialize JSON fields
        item.setdefault('t2i_clip_t2i', None)
        item.setdefault('swap_clip_t2i', None)
        item.setdefault('clip_i2i_score', None)
        item.setdefault('dino_i2i_score', None)
        item.setdefault('t2i_clip_id_i2i', None)
        item.setdefault('swap_clip_id_i2i', None)
        item.setdefault('t2i_dino_id_i2i', None)
        item.setdefault('swap_dino_id_i2i', None)

        # --- 1. Load Images ---
        # A. Swapped Image (傳入 ID 進行比對)
        swapped_path, file_id = smart_find_swapped_image(swapped_dir, prompt, target_id)
        img_swapped = None
        if swapped_path:
            try: img_swapped = Image.open(swapped_path).convert("RGB")
            except: pass
        else:
            missing_swaps.append(f"{target_id} | {prompt[:20]}...")

        # B. T2I Original Image
        t2i_path = find_target_by_prompt(t2i_dir, prompt)
        img_orig = None
        if t2i_path:
            try: img_orig = Image.open(t2i_path).convert("RGB")
            except: pass
            
        # C. Reference Image (Using JSON ID)
        img_ref = None
        if target_id:
            ref_path = smart_find_ref_image(ref_dir, str(target_id).zfill(5))
            if ref_path:
                try: img_ref = Image.open(ref_path).convert("RGB")
                except: pass

        # --- 2. Calculate Metrics ---
        
        # [Metric 1] Prompt Alignment (T2I)
        if prompt:
            if img_orig:
                s = get_clip_t2i_score(clip_model, clip_proc, img_orig, prompt)
                if s is not None:
                    item['t2i_clip_t2i'] = float(f'{s:.4f}')
                    stats['t2i_clip_t2i'].append(s)
            
            if img_swapped:
                s = get_clip_t2i_score(clip_model, clip_proc, img_swapped, prompt)
                if s is not None:
                    item['swap_clip_t2i'] = float(f'{s:.4f}')
                    stats['swap_clip_t2i'].append(s)

        # [Metric 2] Structure Preservation (Orig vs Swapped)
        if img_orig and img_swapped:
            # CLIP Structure
            sc = get_clip_i2i_score(clip_model, clip_proc, img_orig, img_swapped)
            if sc is not None:
                item['clip_i2i_score'] = float(f'{sc:.4f}')
                stats['clip_i2i'].append(sc)
            
            # DINO Structure
            sd = get_dino_score(dino_model, dino_proc, img_orig, img_swapped)
            if sd is not None:
                item['dino_i2i_score'] = float(f'{sd:.4f}')
                stats['dino_i2i'].append(sd)

        # [Metric 3] Identity Similarity (Ref vs T2I / Ref vs Swapped)
        if img_ref:
            # Baseline ID (Ref vs T2I)
            if img_orig:
                # CLIP ID
                cid_t = get_clip_i2i_score(clip_model, clip_proc, img_ref, img_orig)
                if cid_t is not None:
                    item['t2i_clip_id_i2i'] = float(f'{cid_t:.4f}')
                    stats['t2i_clip_id_i2i'].append(cid_t)
                
                # DINO ID
                did_t = get_dino_score(dino_model, dino_proc, img_ref, img_orig)
                if did_t is not None:
                    item['t2i_dino_id_i2i'] = float(f'{did_t:.4f}')
                    stats['t2i_dino_id_i2i'].append(did_t)

            # Swapped ID (Ref vs Swapped)
            if img_swapped:
                # CLIP ID
                cid_s = get_clip_i2i_score(clip_model, clip_proc, img_ref, img_swapped)
                if cid_s is not None:
                    item['swap_clip_id_i2i'] = float(f'{cid_s:.4f}')
                    stats['swap_clip_id_i2i'].append(cid_s)
                
                # DINO ID
                did_s = get_dino_score(dino_model, dino_proc, img_ref, img_swapped)
                if did_s is not None:
                    item['swap_dino_id_i2i'] = float(f'{did_s:.4f}')
                    stats['swap_dino_id_i2i'].append(did_s)

    # Save Results
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    if missing_swaps:
        print(f"\n⚠️  Warning: {len(missing_swaps)} swapped images were not found (ID mismatch).")
        print("   First 5 missing:", missing_swaps[:5])

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    
    return {
        'name': task_name,
        't2i_orig': avg(stats['t2i_clip_t2i']),
        't2i_swap': avg(stats['swap_clip_t2i']),
        'struct_clip': avg(stats['clip_i2i']),
        'struct_dino': avg(stats['dino_i2i']),
        't2i_clip_id_i2i': avg(stats['t2i_clip_id_i2i']),
        'swap_clip_id_i2i': avg(stats['swap_clip_id_i2i']),
        't2i_dino_id_i2i': avg(stats['t2i_dino_id_i2i']),
        'swap_dino_id_i2i': avg(stats['swap_dino_id_i2i']),
        'count': len(stats['swap_clip_id_i2i'])
    }

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch CLIP/DINO Evaluation (Fixed ID Matching)")
    parser.add_argument("--json", type=str, default="metadata.json", help="Source JSON file")
    parser.add_argument("--name", type=str, default="pixart", help="Task Name")
    parser.add_argument("--t2i", type=str, default="/media/ee303/disk2/style_generation/diffusers/pixart_test", help="T2I Dir")
    parser.add_argument("--swap", type=str, default="/media/ee303/disk2/JACK/FACE_SWAPED_pixart_test", help="Swapped Dir")
    parser.add_argument("--ref", type=str, default="/media/ee303/disk2/JACK/reference", help="Reference Dir")
    
    args = parser.parse_args()

    # 1. Load Models
    models = load_models()

    results_summary = []
    print(f"\n📋 Starting Batch Processing (Source: {args.json})...")

    if not os.path.exists(args.swap):
        print(f"⚠️ Skipping {args.name}: Directory not found ({args.swap})")
    else:
        res = process_task(
            args.name, 
            args.swap, 
            args.t2i, 
            args.ref, 
            args.json, 
            models
        )
        if res: results_summary.append(res)

    # 3. Final Summary Table
    print("\n" + "="*145)
    print(f"{'Method':<10} | {'T2I(Orig)':<9} {'T2I(Swap)':<9} | {'Struct(C)':<9} {'Struct(D)':<9} | {'C-ID(Orig)':<10} {'C-ID(Swap)':<10} | {'D-ID(Orig)':<10} {'D-ID(Swap)':<10}")
    print("-" * 145)
    for res in results_summary:
        print(f"{res['name']:<10} | "
              f"{res['t2i_orig']:<9.4f} {res['t2i_swap']:<9.4f} | "
              f"{res['struct_clip']:<9.4f} {res['struct_dino']:<9.4f} | "
              f"{res['t2i_clip_id_i2i']:<10.4f} {res['swap_clip_id_i2i']:<10.4f} | "
              f"{res['t2i_dino_id_i2i']:<10.4f} {res['swap_dino_id_i2i']:<10.4f}")
    print("="*145)
    print("✅ All tasks completed.")