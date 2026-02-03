import sys
import os
import torch
import numpy as np
import argparse
import json
import warnings
from tqdm import tqdm

# 過濾掉包含 "align" 關鍵字的特定警告
warnings.filterwarnings("ignore", message=".*align should be passed as Python.*")

# ==========================================
# 0. AdaFace 模組檢查與路徑設定
# ==========================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AdaFace'))

try:
    import net
    from adaface_alignment import align
except ImportError:
    print("❌ Error: AdaFace module not found. Please ensure 'AdaFace' folder is in the current directory.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. 模型初始化
# ==========================================
def load_adaface_model(architecture='ir_50', model_path="./AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt"):
    print(f"\n🚀 Loading AdaFace ({architecture}) on {DEVICE}...")
    
    if not os.path.exists(model_path):
        alt_path = os.path.join("AdaFace", "pretrained", "adaface_ir50_ms1mv2.ckpt")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"❌ Error: Weights not found at {model_path}")
            sys.exit(1)

    model = net.build_model(architecture)
    statedict = torch.load(model_path, map_location='cpu', weights_only=False)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model.to(DEVICE)
    return model

# ==========================================
# 2. 核心計算函式
# ==========================================
def get_feature(model, img_path):
    """讀取圖片 -> 對齊人臉 -> 提取 AdaFace 特徵"""
    try:
        aligned_rgb_img = align.get_aligned_face(img_path)
        if aligned_rgb_img is None: return None
        
        np_img = np.array(aligned_rgb_img)
        # AdaFace Preprocessing: ((RGB/255) - 0.5) / 0.5 -> BGR Channel
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        bgr_tensor_input = torch.tensor([brg_img.transpose(2,0,1)]).float()
        
        if torch.cuda.is_available():
            bgr_tensor_input = bgr_tensor_input.cuda()
            
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        return feature
    except Exception as e:
        return None

# ==========================================
# 3. 檔案搜尋工具
# ==========================================
def smart_find_swapped_image(base_dir, json_filename):
    if not json_filename: return None, None
    name_no_ext = os.path.splitext(json_filename)[0]
    candidates = [
        f"0_{json_filename}", f"0_{name_no_ext}.png", f"0_{name_no_ext}.jpeg",
        json_filename, f"{name_no_ext}.png"
    ]
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path, cand
    return None, None

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

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 找尋對應的 T2I 原圖"""
    if not prompt or not os.path.exists(base_dir): return None
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if normalize(os.path.splitext(filename)[0]) == target_norm:
            return os.path.join(base_dir, filename)
    return None

# ==========================================
# 4. 單一任務處理邏輯
# ==========================================
def process_task(task_name, swapped_dir, t2i_dir, ref_dir, json_path, model):
    print(f"\n🔹 Processing Task: [{task_name}]")
    print(f"   📂 Swapped Dir: {swapped_dir}")
    print(f"   📂 T2I Source:  {t2i_dir}")
    print(f"   📂 Reference:   {ref_dir}")
    
    if not os.path.exists(json_path):
        print(f"   ❌ JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    scores = {'swapped': [], 't2i': []}
    failed_count = 0

    for item in tqdm(data_list, desc=f"   Running {task_name}"):
        raw_filename = item.get('image', '').strip()
        prompt = item.get('prompt', '').strip()
        
        # 初始化欄位
        item['id_similarity'] = None       # 換臉後的分數
        item['id_similarity_t2i'] = None   # T2I 原圖的分數 (Baseline)

        # 1. 找 Reference Image (所有比較的核心)
        swapped_path, found_name = smart_find_swapped_image(swapped_dir, raw_filename)
        
        # 解析 Ref ID
        ref_idx_str = "0"
        if found_name:
            try:
                fname_no_ext = os.path.splitext(found_name)[0]
                parts = fname_no_ext.split('_')
                ref_idx_str = parts[0] if len(parts) >= 2 else parts[0]
            except: pass
        
        ref_path = smart_find_ref_image(ref_dir, ref_idx_str)
        if not ref_path:
            continue
            
        # 提取 Reference 特徵 (共用)
        feat_ref = get_feature(model, ref_path)
        if feat_ref is None:
            continue

        # --- A. 計算 Swapped Image ID Score ---
        if swapped_path:
            feat_swap = get_feature(model, swapped_path)
            if feat_swap is not None:
                sim_swap = torch.nn.functional.cosine_similarity(feat_ref, feat_swap).item()
                item['id_similarity'] = max(float(f"{sim_swap:.2f}"),0)
                scores['swapped'].append(sim_swap)
            else:
                item['id_similarity'] = 0.0

        # --- B. 計算 T2I Original Image ID Score (Baseline) ---
        t2i_path = find_target_by_prompt(t2i_dir, prompt)
        if t2i_path:
            feat_t2i = get_feature(model, t2i_path)
            if feat_t2i is not None:
                sim_t2i = torch.nn.functional.cosine_similarity(feat_ref, feat_t2i).item()
                item['id_similarity_t2i'] = max(float(f"{sim_t2i:.2f}"),0)
                scores['t2i'].append(sim_t2i)
            else:
                item['id_similarity_t2i'] = 0.0
        
    # 存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    
    return {
        'name': task_name,
        'avg_id_swapped': avg(scores['swapped']),
        'avg_id_t2i': avg(scores['t2i']),
        'count': len(scores['swapped'])
    }

# ==========================================
# 5. 主程式入口
# ==========================================
if __name__ == '__main__':
    # 載入模型
    model = load_adaface_model()

    # ================= 配置區域 (CONFIG) =================
    REFERENCE_DIR = './faceswap_results/reference'
    DEFAULT_T2I_DIR = './pixart_outputs'   # [新增] T2I 原圖路徑
    SOURCE_JSON = 'gt.json'
    
    # 定義實驗
    METHOD_DIRS = {
        'PixArt': './faceswap_results/pixart',
        # 'Janus': './faceswap_results/janus',
        # 'Infinity': './faceswap_results/infinity',
        # 'ShowO2': './faceswap_results/showo2'
    }
    # ====================================================

    results_summary = []
    print(f"\n📋 Starting Batch ID Evaluation (Source: {SOURCE_JSON})...")

    for name, swapped_dir in METHOD_DIRS.items():
        if not os.path.exists(swapped_dir):
            print(f"⚠️ Skipping {name}: Dir not found")
            continue

        res = process_task(
            name, 
            swapped_dir, 
            DEFAULT_T2I_DIR,  # 傳入 T2I 路徑
            REFERENCE_DIR, 
            SOURCE_JSON,
            model
        )
        if res: results_summary.append(res)

    # 最終總表
    print("\n" + "="*80)
    print(f"{'Method':<15} | {'Avg ID (Swapped)':<20} | {'Avg ID (T2I Orig)':<20} | {'Count':<8}")
    print("-" * 80)
    for res in results_summary:
        print(f"{res['name']:<15} | {res['avg_id_swapped']:<20.4f} | {res['avg_id_t2i']:<20.4f} | {res['count']:<8}")
    print("="*80)
    print("✅ All tasks completed.")