import argparse
import json
import os
import sys
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ==========================================
# 0. 環境設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 嘗試匯入專案依賴 (需確保 src 資料夾存在)
try:
    from src.networks import get_EfficientNet_V2
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
except ImportError:
    HAS_REPO_UTILS = False

# 檢查 MediaPipe
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    # 強制匯入 solutions 以支援 Python 3.12+ (如果有的話)
    try:
        import mediapipe.python.solutions as mp_solutions
    except ImportError:
        mp_solutions = mp.solutions
    
    if hasattr(mp_solutions, 'pose'):
        HAS_MEDIAPIPE = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 關鍵點索引
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9 

# ==========================================
# 工具函式: 智慧搜尋圖片
# ==========================================
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
        f"0_{original_filename}",      # 嘗試 0_X.jpg
        original_filename,             # 嘗試 X.jpg (原始)
        f"0_{name_no_ext}.png",        # 嘗試 0_X.png
        f"{name_no_ext}.png"           # 嘗試 X.png
    ]

    for cand in candidates:
        full_path = os.path.join(base_folder, cand)
        if os.path.exists(full_path):
            return full_path, cand
            
    return None, None

# ==========================================
# 1. 核心邏輯 (V-Final-Plus-Plus & Math)
# ==========================================
def normalize_angle(angle):
    if angle is None: return None
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def limit_angle(angle):
    while angle < -180: angle += 360
    while angle > 180: angle -= 360
    return angle

def classify_pose_v_final(b_yaw, b_roll, h_yaw, h_pitch, h_roll, delta):
    """ V-Final-Plus-Plus 分類邏輯 (保留原樣) """
    abs_b_yaw = abs(b_yaw)
    abs_h_yaw = abs(h_yaw)
    
    # 閾值設定
    if b_yaw > 0: THRES_BODY_SIDE_START = 35 
    else: THRES_BODY_SIDE_START = 20

    THRES_BODY_BACK = 89
    THRES_HEAD_FRONT_LIMIT = 30
    THRES_HEAD_PURE_TURN = 22 
    THRES_LEAN = 5 
    THRES_TILT = 8

    # Priority 1: 早期傾斜保護
    if abs_b_yaw < 40 and abs(b_roll) > THRES_LEAN:
        if b_roll > 0: return "Body_Lean_Right"
        else: return "Body_Lean_Left"

    # Priority 2: 背對類
    if abs_b_yaw > THRES_BODY_BACK:
        if abs(delta) < 40: return "Back_View_Straight"
        elif abs_h_yaw < 60: return "Back_Over_Shoulder"
        else: return "Back_View_Side_Looking_Away"

    # Priority 3: 強制頭轉
    if abs_h_yaw > 55 and abs_b_yaw < 60:
         return "Head_Turn_Right" if h_yaw > 0 else "Head_Turn_Left"

    # Priority 4: 側向動作矩陣
    is_body_side = (abs_b_yaw > THRES_BODY_SIDE_START) and (abs_b_yaw <= THRES_BODY_BACK)
    
    if is_body_side:
        final_yaw_direction_sign = 1 if b_yaw > 0 else -1
        if (b_yaw * h_yaw) < 0 and abs_h_yaw > 40:
            final_yaw_direction_sign = 1 if h_yaw > 0 else -1
        
        suffix = "Right" if final_yaw_direction_sign > 0 else "Left"
        is_head_side = abs_h_yaw > THRES_HEAD_FRONT_LIMIT
        
        if not is_head_side:
            return f"Body_Turn_{suffix}_Face_Front"
        else:
            corrected_b_yaw = abs_b_yaw * final_yaw_direction_sign
            if (corrected_b_yaw * h_yaw) > 0: 
                diff = abs_h_yaw - abs_b_yaw
                dominance_gap = 20 if h_yaw > 0 else 6
                if diff > dominance_gap: return f"Head_Turn_{suffix}"
                else: return f"Side_View_{suffix}"
            else: 
                return f"Head_Turn_{suffix}"

    # Priority 5: 純頭轉
    if abs_h_yaw > THRES_HEAD_PURE_TURN:
        return "Head_Turn_Right" if h_yaw > 0 else "Head_Turn_Left"

    # Priority 6: 殘餘歪頭類
    if h_roll > THRES_TILT: return "Head_Tilt_Left"
    if h_roll < -THRES_TILT: return "Head_Tilt_Right"

    # Priority 7: 殘餘傾斜類
    if b_roll > THRES_LEAN: return "Body_Lean_Right"
    if b_roll < -THRES_LEAN: return "Body_Lean_Left"

    # Priority 8: 正面類
    if h_yaw > 15: return "Head_Slight_Right"
    if h_yaw < -15: return "Head_Slight_Left"
    
    return "Frontal"

# ==========================================
# 2. 模型與計算工具
# ==========================================
def load_model_correctly(checkpoint_path):
    print(f"📂 正在解析權重檔: {checkpoint_path}")
    try:
        config = SOTAConfig()
        from src.networks import get_EfficientNet_V2
        model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict_ema', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None

def compute_pose_output(output_tensor):
    if HAS_REPO_UTILS:
        with torch.no_grad():
            rot_mat = batch_torch_A_to_R(output_tensor).cpu().numpy()[0]
    else:
        A = output_tensor.view(-1, 3, 3)
        U, S, V = torch.linalg.svd(A)
        R = torch.matmul(U, V.transpose(1, 2))
        if torch.det(R) < 0:
            V_fixed = V.clone()
            V_fixed[:, :, 2] *= -1
            R = torch.matmul(U, V_fixed.transpose(1, 2))
        rot_mat = R.cpu().numpy()[0]
    rot_mat_2 = np.transpose(rot_mat)
    try:
        r = Rotation.from_matrix(rot_mat_2)
        angles = r.as_euler("xyz", degrees=True)
        return limit_angle(angles[1]), limit_angle(angles[0] - 180), limit_angle(angles[2])
    except:
        return 0.0, 0.0, 0.0

def get_face_box_from_pose(landmarks, w, h):
    x_coords = [landmarks[i].x * w for i in FACE_LANDMARKS_INDICES]
    y_coords = [landmarks[i].y * h for i in FACE_LANDMARKS_INDICES]
    if not x_coords: return None
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    box_size = max(max_x - min_x, max_y - min_y) * 1.5
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [int(cx - box_size/2), int(cy - box_size/2), int(cx + box_size/2), int(cy + box_size/2)]

def calc_body_yaw(landmarks):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return None
    dx, dz = r_sh.x - l_sh.x, r_sh.z - l_sh.z
    return -math.degrees(math.atan2(dz, dx)) * 2.0 

def calc_body_roll(landmarks, width, height):
    l_sh = landmarks[IDX_L_SHOULDER]
    r_sh = landmarks[IDX_R_SHOULDER]
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5: return 0.0
    lx, ly = l_sh.x * width, l_sh.y * height
    rx, ry = r_sh.x * width, r_sh.y * height
    return math.degrees(math.atan2(ly - ry, lx - rx))

# ==========================================
# 3. JSON 處理主程式
# ==========================================
def run_pose_labeling(method, image_dir, json_path, checkpoint_path):
    if not HAS_MEDIAPIPE:
        print("❌ 錯誤: 未安裝 MediaPipe (pip install mediapipe)")
        return

    print(f"Reading JSON from: {json_path}")
    print(f"Image Directory: {image_dir}")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
    
    # 檢查資料夾
    if not os.path.exists(image_dir):
        print(f"⚠️ Warning: Image folder not found at {image_dir}")
        
    # 1. 初始化模型
    mp_pose = mp_solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    head_model = load_model_correctly(checkpoint_path)
    if head_model is None: return

    # 2. 讀取資料
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Total items to process: {len(data_list)}")
    success_count = 0
    missing_count = 0
    
    # 3. 處理迴圈
    for item in tqdm(data_list, desc="Pose Predicting"):
        raw_filename = item.get("image", "").strip()
        
        # === [關鍵修改] 使用智慧搜尋找到正確路徑 ===
        full_image_path, found_name = smart_find_image(image_dir, raw_filename)
        
        prediction = "Image_Not_Found"
        
        if full_image_path:
            try:
                # 圖片處理
                img_pil = Image.open(full_image_path).convert("RGB")
                W, H = img_pil.size
                img_arr = np.array(img_pil)
                results = pose_detector.process(img_arr)
                
                raw_body_yaw = None
                raw_body_roll = 0.0
                h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
                norm_body = None
                
                # 計算特徵
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    raw_body_yaw = calc_body_yaw(lm)
                    raw_body_roll = calc_body_roll(lm, W, H)
                    
                    bbox = get_face_box_from_pose(lm, W, H)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        crop = img_pil.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
                        if crop.size[0] > 5 and crop.size[1] > 5:
                            tf = transforms.Compose([
                                transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            input_t = tf(crop).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                out = head_model(input_t)
                                h_yaw, h_pitch, h_roll = compute_pose_output(out)
                
                # 正規化與預測
                norm_body = normalize_angle(raw_body_yaw)
                norm_h_yaw = normalize_angle(h_yaw)
                
                delta = 0.0
                if norm_body is not None:
                    delta = abs(norm_h_yaw - norm_body)
                    if delta > 180: delta = 360 - delta
                
                if norm_body is None:
                    prediction = "No_Body_Detected"
                else:
                    prediction = classify_pose_v_final(norm_body, raw_body_roll, norm_h_yaw, h_pitch, h_roll, delta)
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {raw_filename}: {e}")
                prediction = "Error"
        else:
            # 真的找不到圖
            missing_count += 1

        # 4. 寫入結果
        item['pose_prediction'] = prediction
        
        # 5. 自動比對正確性
        gt_pose = item.get('gt_pose', '')
        if gt_pose and gt_pose != 'Unknown':
            # 轉成 0 或 1
            item['pose_correct'] = int(prediction == gt_pose)
        else:
            item['pose_correct'] = None

    # 6. 存檔
    print(f"\nSaving updated JSON to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"Success! Processed {success_count} images. Missing {missing_count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart')
    parser.add_argument("--json", type=str, default='output.json')
    parser.add_argument("--checkpoint", default='./pose/checkpoints/SemiUHPE/DAD-WildHead-EffNetV2-S-best.pth')
    args = parser.parse_args()

    # === [關鍵修改] 更新路徑映射 (對應 faceswap_results) ===
    path_map = {
        'pixart': './faceswap_results/pixart',
        'janus': './faceswap_results/janus',
        'infinity': './faceswap_results/infinity',
        'showo2': './faceswap_results/showo2'
    }
    
    # 預設路徑
    image_dir = path_map.get(args.method, './output')

    run_pose_labeling(args.method, image_dir, args.json, args.checkpoint)