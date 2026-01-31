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
# 設定與模型載入
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    print("🚀 Loading CLIP Model (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_dino_model():
    print("🚀 Loading DINOv2 Model (facebook/dinov2-base)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    return model, processor

# ==========================================
# 計算函式
# ==========================================

# 1. CLIP Text-to-Image (T2I)
def get_clip_t2i_score(model, processor, image, text):
    try:
        inputs = processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        score = (image_embeds @ text_embeds.t()).item()
        return max(0.0, score)
    except Exception as e:
        print(f"[CLIP T2I Error] {e}")
        return None

# 2. CLIP Image-to-Image (I2I)
def get_clip_i2i_score(model, processor, img_ref, img_gen):
    try:
        inputs = processor(images=[img_ref, img_gen], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        ref_embed = image_features[0].unsqueeze(0)
        gen_embed = image_features[1].unsqueeze(0)

        ref_embed = ref_embed / ref_embed.norm(p=2, dim=-1, keepdim=True)
        gen_embed = gen_embed / gen_embed.norm(p=2, dim=-1, keepdim=True)

        score = (ref_embed @ gen_embed.t()).item()
        return max(0.0, score)
    except Exception as e:
        print(f"[CLIP I2I Error] {e}")
        return None

# 3. DINO Image-to-Image
def get_dino_score(model, processor, img_ref, img_gen):
    try:
        inputs1 = processor(images=img_ref, return_tensors="pt").to(DEVICE)
        inputs2 = processor(images=img_gen, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out1 = model(**inputs1).last_hidden_state[:, 0, :]
            out2 = model(**inputs2).last_hidden_state[:, 0, :]
            
        score = F.cosine_similarity(out1, out2).item()
        return max(0.0, score)
    except Exception as e:
        print(f"[DINO Error] {e}")
        return None

# ==========================================
# 智慧路徑搜尋
# ==========================================
def smart_find_swapped_image(base_dir, json_filename):
    name_no_ext = os.path.splitext(json_filename)[0]
    candidates = [
        f"0_{json_filename}",           # 0_1.jpg
        f"0_{name_no_ext}.png",         # 0_1.png
        json_filename,                  # 1.jpg
        f"{name_no_ext}.png"            # 1.png
    ]
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path, cand
    return None, None

def find_target_by_prompt(base_dir, prompt):
    if not prompt: return None
    def normalize_quotes(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_normalized = normalize_quotes(prompt)
    
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            file_no_ext = os.path.splitext(filename)[0]
            if normalize_quotes(file_no_ext) == target_normalized:
                return os.path.join(base_dir, filename)
    return None

# ==========================================
# 主程式
# ==========================================
def main(method, swapped_dir, t2i_dir, json_path): 
    """
    swapped_dir: 換臉後的圖片資料夾
    t2i_dir: 原始 T2I 生成的圖片資料夾 (用於比較結構與原始 Prompt Alignment)
    """
    print(f"📂 Swapped Dir: {swapped_dir}")
    print(f"📂 T2I Source Dir: {t2i_dir}")
    
    # 1. 載入模型
    clip_model, clip_proc = load_clip_model()
    dino_model, dino_proc = load_dino_model()
    
    # 2. 讀取 JSON
    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        
    print(f"📊 Processing {len(data_list)} items...")
    
    # --- [MODIFIED] 統計用列表 (分離不同指標) ---
    stats = {
        'clip_t2i_orig': [],    # 原始 T2I 圖 vs Prompt
        'clip_t2i_swap': [],    # 換臉後圖 vs Prompt
        'clip_struct': [],      # 原始 T2I 圖 vs 換臉後圖 (CLIP)
        'dino_struct': []       # 原始 T2I 圖 vs 換臉後圖 (DINO)
    }
    
    for item in tqdm(data_list, desc="Calculating Metrics"):
        raw_filename = item.get('image', '').strip()
        prompt = item.get('prompt', '').strip()
        
        # --- A. 找生成圖 (Swapped Image) ---
        swapped_path, found_name = smart_find_swapped_image(swapped_dir, raw_filename)
        
        # 初始化欄位為 None
        item['clip_t2i_orig_score'] = None
        item['clip_t2i_swapped_score'] = None
        item['clip_score'] = None
        item['dino_score'] = None

        if not swapped_path:
            continue
            
        try:
            img_swapped = Image.open(swapped_path).convert("RGB") # swapped image
        except:
            continue

        # --- B. 找原始 T2I 圖 (Original Generated Image) ---
        target_path = find_target_by_prompt(t2i_dir, prompt)
        if not target_path:
            # 如果找不到原圖，只能算 Swapped 的 T2I 分數，無法算結構分
            if prompt:
                t2i_swapped_score = get_clip_t2i_score(clip_model, clip_proc, img_swapped, prompt)
                item['clip_t2i_swapped_score'] = float(f'{t2i_swapped_score:.4f}')
                if t2i_swapped_score is not None: stats['clip_t2i_swap'].append(t2i_swapped_score)
            continue
            
        try:
            img_orig_t2i = Image.open(target_path).convert("RGB") # t2i generated image
        except:
            continue

        # --- C. 計算 CLIP T2I Score (Prompt Alignment) ---
        # 1. 原始 T2I 圖 vs Prompt
        if prompt:
            t2i_orig_score = get_clip_t2i_score(clip_model, clip_proc, img_orig_t2i, prompt)
            item['clip_t2i_orig_score'] = float(f'{t2i_orig_score:.2f}')
            if t2i_orig_score is not None: stats['clip_t2i_orig'].append(t2i_orig_score)
            
            # 2. 換臉後圖 vs Prompt
            t2i_swapped_score = get_clip_t2i_score(clip_model, clip_proc, img_swapped, prompt)
            item['clip_t2i_swapped_score'] = float(f'{t2i_swapped_score:.2f}')
            if t2i_swapped_score is not None: stats['clip_t2i_swap'].append(t2i_swapped_score)

        # --- D. 計算 Image-to-Image 分數 (Structure Preservation) ---
        # 比較對象：原始 T2I 圖 vs 換臉後圖 (衡量背景與構圖是否改變)
        try:
            # 1. CLIP I2I (Structure)
            c_struct_score = get_clip_i2i_score(clip_model, clip_proc, img_orig_t2i, img_swapped)
            item['clip_score'] = float(f'{c_struct_score:.2f}')
            if c_struct_score is not None: stats['clip_struct'].append(c_struct_score)

            # 2. DINO Score (Structure)
            d_struct_score = get_dino_score(dino_model, dino_proc, img_orig_t2i, img_swapped)
            item['dino_score'] = float(f'{d_struct_score:.2f}')
            if d_struct_score is not None: stats['dino_struct'].append(d_struct_score)

        except Exception as e:
            print(f"Error processing Structure I2I: {e}")


    # 3. 存檔與顯示結果
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Done! Updated JSON saved to {json_path}")
    
    # --- [MODIFIED] Summary ---
    print("\n" + "=" * 60)
    print("🏆 Performance Summary (Averages)")
    print("=" * 60)
    
    def calc_avg(lst):
        return sum(lst)/len(lst) if lst else 0.0

    print(f"📝 Prompt Alignment (Original T2I):  {calc_avg(stats['clip_t2i_orig']):.4f}")
    print(f"📝 Prompt Alignment (Swapped Face):  {calc_avg(stats['clip_t2i_swap']):.4f}")
    print("-" * 60)
    print(f"🏗️  Structure Preservation (CLIP):    {calc_avg(stats['clip_struct']):.4f} (Orig vs Swapped)")
    print(f"🦖 Structure Preservation (DINO):    {calc_avg(stats['dino_struct']):.4f} (Orig vs Swapped)")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart')
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
    args = parser.parse_args()

    # 路徑映射
    path_map = {
        'pixart': './faceswap_results/pixart',
        'janus': './faceswap_results/janus',
        'infinity': './faceswap_results/infinity',
        'showo2': './faceswap_results/showo2'
    }
    t2i_dir = './pixart_outputs' 
    swapped_dir = path_map.get(args.method, './faceswap_results/pixart')
    
    reference_dir = './faceswap_results/reference' 

    main(args.method, swapped_dir, t2i_dir, args.json)