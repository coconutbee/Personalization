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
    print("❌ Error: AdaFace module not found.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. 模型初始化與特徵提取 (保持不變)
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
def smart_find_ref_image(ref_dir, rand_id):
    if not os.path.exists(ref_dir): return None
    for filename in os.listdir(ref_dir):
        if filename.startswith(f"{rand_id}_") or filename == f"{rand_id}.png":
            return os.path.join(ref_dir, filename)
    return None

def find_target_by_prompt(base_dir, prompt):
    if not prompt or not os.path.exists(base_dir): return None
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip().lower()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if normalize(os.path.splitext(filename)[0]) == target_norm:
            return os.path.join(base_dir, filename)
    return None

# ==========================================
# 3. 核心處理邏輯 (數據合併模式)
# ==========================================
def process_task_merge(task_name, swapped_dir, t2i_dir, ref_dir, json_path, model):
    print(f"\n🔹 Processing Task (ID Similarity): [{task_name}]")
    
    # A. 優先掃描 Swap 資料夾取得 100 張圖的路徑
    swap_files = [f for f in os.listdir(swapped_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not swap_files:
        print(f"❌ Swapped 資料夾中找不到圖片: {swapped_dir}")
        return None

    # B. 嘗試讀取現有的 JSON (保留 Pose 等其他結果)
    existing_data_map = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                raw_json = json.load(f)
                # 以檔名作為 Key 建立暫時索引
                existing_data_map = {item['image']: item for item in raw_json if 'image' in item}
                print(f"📂 載入現有 JSON，發現 {len(existing_data_map)} 筆已存在的紀錄。")
            except Exception as e:
                print(f"⚠️ JSON 格式有誤或為空，將重新建立: {e}")

    results_list = []
    scores_swapped = []
    t2i_baseline_cache = {}

    print(f"🚀 開始計算 ID 相似度 (目標: {len(swap_files)} 張圖)...")
    for filename in tqdm(swap_files, desc=f"   Running {task_name}"):
        if "_" not in filename: continue
        
        # 解析檔名
        parts = filename.split('_', 1)
        rand_id = parts[0]
        prompt = os.path.splitext(parts[1])[0]
        
        swapped_path = os.path.join(swapped_dir, filename)
        ref_path = smart_find_ref_image(ref_dir, rand_id)
        t2i_path = find_target_by_prompt(t2i_dir, prompt)

        if not ref_path: continue

        # 提取與計算
        feat_ref = get_feature(model, ref_path)
        feat_swap = get_feature(model, swapped_path)
        
        sim_swap = 0.0
        if feat_ref is not None and feat_swap is not None:
            sim_swap = torch.nn.functional.cosine_similarity(feat_ref, feat_swap).item()
            scores_swapped.append(sim_swap)

        # Baseline 計算與快取
        cache_key = f"{rand_id}_{prompt}"
        if cache_key not in t2i_baseline_cache:
            feat_t2i = get_feature(model, t2i_path) if t2i_path else None
            t2i_baseline_cache[cache_key] = torch.nn.functional.cosine_similarity(feat_ref, feat_t2i).item() if feat_t2i is not None else 0.0
        sim_t2i = t2i_baseline_cache[cache_key]

        # --- 合併關鍵：保留舊數據，新增/更新 ID 分數 ---
        # 如果現有 JSON 裡已經有這張圖的 Pose 分數，我們會拿出來用
        item_data = existing_data_map.get(filename, {
            "rand_id": rand_id,
            "image": filename,
            "prompt": prompt
        })
        
        item_data.update({
            "swap_id_similarity": round(max(sim_swap, 0), 2),
            "t2i_id_similarity": round(max(sim_t2i, 0), 2)
        })
        results_list.append(item_data)

    # 寫回 JSON (合併完成後儲存)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    return {
        'name': task_name,
        'avg_id_swapped': avg(scores_swapped),
        'avg_id_t2i': avg(list(t2i_baseline_cache.values())),
        'count_swapped': len(scores_swapped),
        'count_t2i': len(t2i_baseline_cache)
    }

# ==========================================
# 5. 主程式入口 (參數與您指定的一致)
# ==========================================
if __name__ == '__main__':
    model = load_adaface_model()
    
    parser = argparse.ArgumentParser(description="Batch AdaFace ID Evaluation with Merge Support")
    parser.add_argument("--json", type=str, required=True, help="JSON 檔案路徑")
    parser.add_argument("--name", type=str, required=True, help="任務名稱")
    parser.add_argument("--t2i", type=str, required=True, help="T2I 生成圖片資料夾路徑")
    parser.add_argument("--swap", type=str, required=True, help="Swap 生成圖片資料夾路徑")
    parser.add_argument("--ref", type=str, required=True, help="Reference 參考圖片資料夾路徑")
    args = parser.parse_args()

    res = process_task_merge(args.name, args.swap, args.t2i, args.ref, args.json, model)

    if res:
        print("\n" + "="*100)
        print(f"{res['name']:<15} | {res['avg_id_swapped']:<20.4f} | {res['avg_id_t2i']:<20.4f} | {res['count_swapped']}/{res['count_t2i']}")
        print("="*100)
        print(f"✅ 計算完成！結果已合併至: {args.json}")