import argparse
import json
import os
import sys
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

# -----------------------------
# 0. 環境設定 & MiVOLO 初始化
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print("Initializing MiVOLO + InsightFace ...")

# 初始化 FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 初始化 MiVOLO
mivolo_model_id = "iitolstykh/mivolo_v2"
try:
    mivolo_cfg = AutoConfig.from_pretrained(mivolo_model_id, trust_remote_code=True)
    mivolo_model = AutoModelForImageClassification.from_pretrained(
        mivolo_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    mivolo_processor = AutoImageProcessor.from_pretrained(mivolo_model_id, trust_remote_code=True)
except Exception as e:
    print(f"Error loading MiVOLO: {e}")
    sys.exit(1)

# -----------------------------
# 工具函式: 智慧搜尋圖片
# -----------------------------
def smart_find_image(base_folder, original_filename):
    """
    自動嘗試多種檔名組合，直到找到存在的檔案
    回傳: (full_path, found_filename) 或 (None, None)
    """
    if not original_filename:
        return None, None

    # 取得檔名與副檔名
    name_no_ext, ext = os.path.splitext(original_filename)
    
    # 定義所有可能的候選檔名
    candidates = [
        f"0_{original_filename}",      # 嘗試 0_X.jpg (例如 0_1.jpg)
        original_filename,             # 嘗試 X.jpg (例如 1.jpg)
        f"0_{name_no_ext}.png",        # 嘗試 0_X.png
        f"{name_no_ext}.png"           # 嘗試 X.png
    ]

    for cand in candidates:
        full_path = os.path.join(base_folder, cand)
        if os.path.exists(full_path):
            return full_path, cand
            
    return None, None

# -----------------------------
# 1. 預測核心函式
# -----------------------------
def mivolo_predict(image_path):
    """
    讀取圖片 -> 偵測人臉 -> 裁切 -> MiVOLO 預測
    Return: (age, gender_label)
    """
    if not image_path or not os.path.isfile(image_path):
        return None, None

    # 使用 OpenCV 讀取
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. 人臉偵測
    faces = face_app.get(img)
    
    # 2. 裁切人臉
    if len(faces) == 0:
        crop = img
    else:
        face = faces[0]
        h, w = img.shape[:2]
        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1: 
                crop = img
            else: 
                crop = img[y1:y2, x1:x2]
        except:
            crop = img

    if crop is None or crop.size == 0:
        crop = img

    # 3. MiVOLO 推論
    try:
        inputs = mivolo_processor(images=[crop])["pixel_values"]
        inputs = inputs.to(dtype=mivolo_model.dtype, device=mivolo_model.device)
        dummy_body = torch.zeros_like(inputs)

        with torch.no_grad():
            output = mivolo_model(faces_input=inputs, body_input=dummy_body)

        age = float(output.age_output[0].item())
        gender_idx = int(output.gender_class_idx[0].item())
        gender_label = mivolo_cfg.gender_id2label[gender_idx].lower() 
        return age, gender_label
        
    except Exception as e:
        print(f"Prediction error for {image_path}: {e}")
        return None, None

# -----------------------------
# 2. JSON 處理主流程
# -----------------------------
def run_gender_labeling(method, base_folder_path, json_path):
    print(f"Reading JSON from: {json_path}")
    print(f"Image Directory: {base_folder_path}")

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
    
    # 檢查圖片資料夾是否存在
    if not os.path.exists(base_folder_path):
        print(f"⚠️ Warning: Image folder not found at {base_folder_path}")

    # 讀取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Total items to process: {len(data_list)}")
    
    success_count = 0
    missing_count = 0

    # 開始迴圈
    for item in tqdm(data_list, desc="MiVOLO Predicting"):
        raw_filename = item.get("image", "").strip()
        
        # === [關鍵修改] 使用智慧搜尋找到正確路徑 ===
        full_image_path, found_name = smart_find_image(base_folder_path, raw_filename)
        
        # 如果找不到圖
        if not full_image_path:
            item['mivolo_gender'] = "image_not_found"
            item['mivolo_age'] = None
            item['gender_correct'] = 0
            missing_count += 1
            continue

        # 執行預測
        pred_age, pred_gender = mivolo_predict(full_image_path)
        
        if pred_age is not None:
            # 寫入預測結果
            item['mivolo_gender'] = pred_gender
            item['mivolo_age'] = round(pred_age, 1)
            
            # 自動比對 Ground Truth
            gt_gender = item.get('gt_gender', '').lower()
            if gt_gender and gt_gender != 'unknown':
                # 轉成 0 或 1
                is_correct = 1 if (gt_gender == pred_gender) else 0
                item['gender_correct'] = is_correct
            else:
                item['gender_correct'] = None
            
            success_count += 1
        else:
            # 預測過程失敗 (例如無法裁切)
            item['mivolo_gender'] = "prediction_failed"
            item['mivolo_age'] = None
            item['gender_correct'] = 0
            missing_count += 1

    # 存檔
    print(f"\nSaving updated JSON to: {json_path}")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print("Success!")
        print(f"Processed: {success_count}, Missing/Error: {missing_count}")
        
    except Exception as e:
        print(f"Error saving JSON: {e}")

# -----------------------------
# 3. 程式入口
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart', help="Name of the generation method")
    parser.add_argument("--json", type=str, default='output.json', help="Path to the JSON file")
    args = parser.parse_args()
    
    # === [關鍵修改] 更新路徑映射 (對應 faceswap_results) ===
    path_map = {
        'pixart': './faceswap_results/pixart',
        'janus': './faceswap_results/janus',
        'infinity': './faceswap_results/infinity',
        'showo2': './faceswap_results/showo2'
    }
    
    # 預設路徑
    base_folder_path = path_map.get(args.method, './output')

    run_gender_labeling(args.method, base_folder_path, args.json)