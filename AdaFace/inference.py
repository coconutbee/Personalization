import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse
import json
import warnings
from tqdm import tqdm

# 過濾掉包含 "align" 關鍵字的特定警告
warnings.filterwarnings("ignore", message=".*align should be passed as Python.*")

# --- Import AdaFace ---
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AdaFace'))
try:
    import net
    from adaface_alignment import align
except ImportError:
    print("❌ 錯誤: 找不到 AdaFace 模組，請確認資料夾結構。")
    sys.exit(1)

# --- 模型設定 ---
adaface_models = {
    'ir_50': "./AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    if not os.path.exists(adaface_models[architecture]):
        print(f"❌ 錯誤: 找不到權重檔 {adaface_models[architecture]}")
        sys.exit(1)
        
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location='cpu', weights_only=False)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def get_feature(model, img_path):
    try:
        aligned_rgb_img = align.get_aligned_face(img_path)
        if aligned_rgb_img is None: return None
        
        # 轉 tensor
        np_img = np.array(aligned_rgb_img)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        bgr_tensor_input = torch.tensor([brg_img.transpose(2,0,1)]).float()
        
        if torch.cuda.is_available():
            bgr_tensor_input = bgr_tensor_input.cuda()
            
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        return feature
    except Exception as e:
        print(f"[Error] Feature extraction failed for {os.path.basename(img_path)}: {e}")
        return None

# --- [關鍵修改] 智慧搜尋 Swapped 圖片 ---
def smart_find_swapped_image(base_dir, json_filename):
    """
    根據 JSON 檔名 (e.g. '1.jpg') 尋找磁碟上的檔案 (e.g. '0_1.png')
    """
    name_no_ext = os.path.splitext(json_filename)[0] # "1"
    
    # 定義搜尋清單 (優先順序)
    candidates = [
        f"0_{json_filename}",           # 0_1.jpg
        f"0_{name_no_ext}.png",         # 0_1.png
        f"0_{name_no_ext}.jpeg",        # 0_1.jpeg
        json_filename,                  # 1.jpg (備用)
        f"{name_no_ext}.png"            # 1.png (備用)
    ]
    
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path, cand
            
    return None, candidates  # 回傳嘗試過的列表以便除錯

# --- [工具] 智慧搜尋 Reference 圖片 ---
def smart_find_ref_image(ref_dir, ref_index_str):
    candidates = [
        f"{ref_index_str}.jpg",
        f"{ref_index_str}.png",
        f"{ref_index_str}.jpeg",
        f"0_{ref_index_str}.jpg"  # 預防 Reference 自己也有前綴
    ]
    for cand in candidates:
        full_path = os.path.join(ref_dir, cand)
        if os.path.exists(full_path):
            return full_path
    return None

def run_id_evaluation(method, swapped_dir, reference_dir, json_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading AdaFace on {device}...")
    model = load_pretrained_model('ir_50')
    model.to(device)

    print(f"📂 Swapped Dir: {os.path.abspath(swapped_dir)}")
    print(f"📂 Reference Dir: {os.path.abspath(reference_dir)}")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"📊 Processing {len(data_list)} items...")
    
    success_count = 0
    total_score = 0.0
    valid_pairs = 0
    
    # 用來檢查是否所有路徑都錯誤 (如果是，印出詳細除錯資訊)
    failed_logs = []

    for item in tqdm(data_list, desc="ID Similarity"):
        raw_filename = item.get('image', '').strip() # JSON: "1.jpg"
        if not raw_filename: continue

        # 1. 找 Swapped Image
        swapped_path, found_name = smart_find_swapped_image(swapped_dir, raw_filename)
        
        if not swapped_path:
            # 記錄失敗原因
            _, tried_list = smart_find_swapped_image(swapped_dir, raw_filename) # 重新取得列表
            failed_logs.append(f"JSON Image '{raw_filename}' -> Tried {tried_list} in {swapped_dir} -> NOT FOUND")
            item['id_similarity'] = None
            continue

        # 2. 解析 Reference ID
        # 檔名通常是 "0_1.jpg" -> Ref="0"
        try:
            fname_no_ext = os.path.splitext(found_name)[0]
            parts = fname_no_ext.split('_')
            
            if len(parts) >= 2:
                ref_idx_str = parts[0] # 取第一個數字作為 Reference ID
            else:
                # 如果檔名沒有底線 (e.g. "1.jpg")，假設 Ref ID = 1
                ref_idx_str = parts[0]
        except:
            ref_idx_str = "0" # fallback

        # 3. 找 Reference Image
        ref_path = smart_find_ref_image(reference_dir, ref_idx_str)
        if not ref_path:
            failed_logs.append(f"Ref ID '{ref_idx_str}' -> Looked in {reference_dir} -> NOT FOUND")
            item['id_similarity'] = None
            continue

        # 4. 提取特徵 & 計算
        feat_ref = get_feature(model, ref_path)
        feat_swap = get_feature(model, swapped_path)

        if feat_ref is not None and feat_swap is not None:
            similarity = torch.nn.functional.cosine_similarity(feat_ref, feat_swap).item()
            item['id_similarity'] = float(f"{similarity:.2f}")
            total_score += similarity
            valid_pairs += 1
            success_count += 1
        else:
            item['id_similarity'] = None
            if feat_ref is None: failed_logs.append(f"Face detect fail: Ref {os.path.basename(ref_path)}")
            if feat_swap is None: failed_logs.append(f"Face detect fail: Swap {found_name}")

    # 5. 輸出 Debug 資訊 (如果失敗太多)
    if valid_pairs == 0 and len(failed_logs) > 0:
        print("\n⚠️  [DEBUG REPORT] No valid pairs found. Here is why:")
        for log in failed_logs[:5]: # 只印前 5 條避免洗版
            print(f"   ❌ {log}")
        print("   (Check your paths and file extensions!)")

    # 存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Done! Updated JSON saved to {json_path}")
    
    if valid_pairs > 0:
        avg_sim = total_score / valid_pairs
        print(f"🧠 Average ID Similarity: {avg_sim:.4f} (Calculated from {valid_pairs} pairs)")

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

    run_id_evaluation(args.method, swapped_dir, reference_dir, args.json)