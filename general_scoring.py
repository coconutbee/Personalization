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

# 2. CLIP Image-to-Image (I2I) [新增]
def get_clip_i2i_score(model, processor, img_ref, img_gen):
    try:
        # 同時輸入兩張圖提取特徵
        inputs = processor(images=[img_ref, img_gen], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # 使用 get_image_features 只跑 Vision Encoder
            image_features = model.get_image_features(**inputs)
        
        # 分別取出 Ref 和 Gen 的特徵
        ref_embed = image_features[0].unsqueeze(0)
        gen_embed = image_features[1].unsqueeze(0)

        # Normalize
        ref_embed = ref_embed / ref_embed.norm(p=2, dim=-1, keepdim=True)
        gen_embed = gen_embed / gen_embed.norm(p=2, dim=-1, keepdim=True)

        # Cosine Similarity
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

def smart_find_ref_image(ref_dir, ref_index_str):
    candidates = [
        f"{ref_index_str}.jpg",
        f"{ref_index_str}.png",
        f"{ref_index_str}.jpeg",
        f"0_{ref_index_str}.jpg"
    ]
    for cand in candidates:
        full_path = os.path.join(ref_dir, cand)
        if os.path.exists(full_path):
            return full_path
    return None

# ==========================================
# 主程式
# ==========================================
def main(method, swapped_dir, reference_dir, json_path): 
    """
    swapped_dir: 用換臉後的圖片資料夾
    reference_dir: 與ID對應的參考圖片資料夾
    """
    print(f"📂 Swapped Dir: {swapped_dir}")
    print(f"📂 Reference Dir: {reference_dir}")
    
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
    
    # 統計用列表
    stats = {
        'clip_t2i': [],
        'clip_i2i': [],
        'dino': []
    }
    
    for item in tqdm(data_list, desc="Calculating Metrics"):
        raw_filename = item.get('image', '').strip()
        prompt = item.get('prompt', '').strip()
        
        # --- A. 找生成圖 (Swapped Image) ---
        swapped_path, found_name = smart_find_swapped_image(swapped_dir, raw_filename)
        
        if not swapped_path:
            item['clip_t2i_score'] = None
            item['clip_i2i_score'] = None
            item['dino_score'] = None
            continue
            
        try:
            img_gen = Image.open(swapped_path).convert("RGB")
        except:
            continue

        # --- B. 計算 CLIP T2I Score (Text vs Generated Image) ---
        if prompt:
            t2i_score = get_clip_t2i_score(clip_model, clip_proc, img_gen, prompt)
            item['clip_t2i_score'] = float(f'{t2i_score:.2f}')
            if t2i_score is not None: stats['clip_t2i'].append(t2i_score)
        else:
            item['clip_t2i_score'] = None

        # --- C. 找參考圖 (Reference Image) ---
        try:
            fname_no_ext = os.path.splitext(found_name)[0]
            parts = fname_no_ext.split('_')
            ref_idx_str = parts[0] if len(parts) >= 2 else parts[0]
        except:
            ref_idx_str = "0"

        ref_path = smart_find_ref_image(reference_dir, ref_idx_str)
        
        # --- D. 計算 Image-to-Image 分數 (Reference vs Generated) ---
        if ref_path:
            try:
                img_ref = Image.open(ref_path).convert("RGB")
                
                # 1. CLIP I2I
                i2i_score = get_clip_i2i_score(clip_model, clip_proc, img_ref, img_gen)
                item['clip_i2i_score'] = float(f'{i2i_score:.2f}')
                if i2i_score is not None: stats['clip_i2i'].append(i2i_score)

                # 2. DINO Score
                d_score = get_dino_score(dino_model, dino_proc, img_ref, img_gen)
                item['dino_score'] = float(f'{d_score:.2f}')
                if d_score is not None: stats['dino'].append(d_score)

            except Exception as e:
                print(f"Error processing I2I: {e}")
                item['clip_i2i_score'] = None
                item['dino_score'] = None
        else:
            item['clip_i2i_score'] = None
            item['dino_score'] = None

    # 3. 存檔與顯示結果
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Done! Updated JSON saved to {json_path}")
    
    print("-" * 50)
    print("🏆 Performance Summary")
    print("-" * 50)
    if stats['clip_t2i']:
        print(f"📝 Avg CLIP T2I Score (Text-Gen):  {sum(stats['clip_t2i'])/len(stats['clip_t2i']):.4f}")
    if stats['clip_i2i']:
        print(f"🖼️  Avg CLIP I2I Score (Ref-Gen):   {sum(stats['clip_i2i'])/len(stats['clip_i2i']):.4f}")
    if stats['dino']:
        print(f"🦖 Avg DINO Score (Ref-Gen):       {sum(stats['dino'])/len(stats['dino']):.4f}")
    print("-" * 50)

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
    
    swapped_dir = path_map.get(args.method, './faceswap_results/pixart')
    reference_dir = './faceswap_results/reference' 

    main(args.method, swapped_dir, reference_dir, args.json)