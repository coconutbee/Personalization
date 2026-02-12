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
# 假設 AdaFace 資料夾與此腳本在同一目錄下
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AdaFace'))

try:
    import net
    from adaface_alignment import align
except ImportError:
    print("❌ Error: AdaFace module not found. Please ensure the 'AdaFace' folder is present.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. 模型初始化與特徵提取
# ==========================================
def load_adaface_model(architecture='ir_50', model_path="./AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt"):
    print(f"\n🚀 Loading AdaFace ({architecture}) on {DEVICE}...")
    model = net.build_model(architecture)
    statedict = torch.load(model_path, map_location='cpu', weights_only=False)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model.to(DEVICE)
    return model

def get_feature(model, img_path):
    try:
        aligned_rgb_img = align.get_aligned_face(img_path)
        if aligned_rgb_img is None: return None
        np_img = np.array(aligned_rgb_img)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        bgr_tensor_input = torch.tensor([brg_img.transpose(2,0,1)]).float().to(DEVICE)
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        return feature
    except Exception: return None

# ==========================================
# 2. 搜尋工具
# ==========================================
def find_ref_image_by_id(ref_dir, ref_id_str):
    """
    根據 ID (如 '00051') 找尋 Reference 圖片。
    假設 Reference 檔名可能是 '00051.jpg' 或 '00051_xxx.jpg'。
    """
    if not os.path.exists(ref_dir): return None
    
    # 優先找完全匹配的 ID
    for ext in ['.jpg', '.png', '.jpeg', '.webp']:
        exact_path = os.path.join(ref_dir, f"{ref_id_str}{ext}")
        if os.path.exists(exact_path):
            return exact_path
            
    # 如果找不到，嘗試找前綴匹配 (例如 00051_face.jpg)
    for filename in os.listdir(ref_dir):
        if filename.startswith(f"{ref_id_str}_") or filename.startswith(f"{ref_id_str}."):
             return os.path.join(ref_dir, filename)
             
    return None

# ==========================================
# 3. 核心處理邏輯 (JSON 驅動模式)
# ==========================================
def process_t2i_id_similarity(task_name, t2i_dir, ref_dir, json_path, model):
    print(f"\n🔹 Processing Task (ID Similarity - T2I vs Ref): [{task_name}]")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return None

    # 讀取 JSON 資料
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        
    print(f"📂 Loaded JSON with {len(data_list)} items.")

    scores_t2i = []
    processed_count = 0
    
    # 用來快取 Reference 的特徵，避免重複計算 (因為同一個 ID 可能有多個 Prompt)
    ref_feature_cache = {} 

    print(f"🚀 Starting ID Similarity Calculation...")
    
    for item in tqdm(data_list, desc=f"   Running {task_name}"):
        # 從 JSON 取得資訊
        img_filename = item.get('image')     # e.g., "00051_A_cute_cat.jpg"
        ref_id = str(item.get('id'))         # e.g., "00051"
        
        if not img_filename or not ref_id: continue

        # 1. 取得 T2I 圖片路徑
        t2i_path = os.path.join(t2i_dir, img_filename)
        if not os.path.exists(t2i_path):
            # print(f"⚠️ Warning: T2I image not found: {t2i_path}")
            item['t2i_id_similarity'] = 0.0
            continue

        # 2. 取得 Reference 圖片路徑
        ref_path = find_ref_image_by_id(ref_dir, ref_id)
        if not ref_path:
            # print(f"⚠️ Warning: Reference image for ID {ref_id} not found in {ref_dir}")
            item['t2i_id_similarity'] = 0.0
            continue

        # 3. 計算特徵 (Reference 使用 Cache)
        if ref_id in ref_feature_cache:
            feat_ref = ref_feature_cache[ref_id]
        else:
            feat_ref = get_feature(model, ref_path)
            ref_feature_cache[ref_id] = feat_ref # 即使是 None 也存起來，避免重複嘗試無效圖片

        feat_t2i = get_feature(model, t2i_path)

        # 4. 計算 Cosine Similarity
        sim_t2i = 0.0
        if feat_ref is not None and feat_t2i is not None:
            sim_t2i = torch.nn.functional.cosine_similarity(feat_ref, feat_t2i).item()
            # 限制在 0~1 之間 (雖然 cosine 可能負數，但人臉相似度通常看正值)
            sim_t2i = max(sim_t2i, 0.0)
            scores_t2i.append(sim_t2i)
        
        # 5. 更新 JSON item
        item['t2i_id_similarity'] = round(sim_t2i, 4)
        processed_count += 1

    # 寫回 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    
    return {
        'name': task_name,
        'avg_id_t2i': avg(scores_t2i),
        'count': len(scores_t2i),
        'total_processed': processed_count
    }

# ==========================================
# 4. 主程式入口
# ==========================================
if __name__ == '__main__':
    # 初始化模型
    model = load_adaface_model()
    
    parser = argparse.ArgumentParser(description="AdaFace ID Similarity for T2I (JSON Driven)")
    parser.add_argument("--json", type=str, required=True, help="Input/Output JSON file path")
    parser.add_argument("--t2i", type=str, required=True, help="Folder containing T2I images")
    parser.add_argument("--ref", type=str, required=True, help="Folder containing Reference images")
    parser.add_argument("--name", type=str, default="t2i_eval", help="Task Name")
    
    args = parser.parse_args()

    # 執行計算
    res = process_t2i_id_similarity(args.name, args.t2i, args.ref, args.json, model)

    if res:
        print("\n" + "="*80)
        print(f"Task: {res['name']}")
        print(f"Average ID Similarity (T2I vs Ref): {res['avg_id_t2i']:.4f}")
        print(f"Valid Comparisons: {res['count']} / {res['total_processed']}")
        print("="*80)
        print(f"✅ Results updated in: {args.json}")