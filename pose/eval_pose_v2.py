import argparse
import json
import os
import sys
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import joblib 
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ==========================================
# 0. 環境設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 嘗試匯入專案依賴
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
# 工具函式
# ==========================================
def smart_find_image(base_folder, original_filename):
    """ 自動嘗試多種檔名組合 """
    if not original_filename:
        return None, None
    name_no_ext, ext = os.path.splitext(original_filename)
    candidates = [
        f"0_{original_filename}", 
        original_filename, 
        f"0_{name_no_ext}.png", 
        f"{name_no_ext}.png"
    ]
    for cand in candidates:
        full_path = os.path.join(base_folder, cand)
        if os.path.exists(full_path):
            return full_path, cand
    return None, None

def find_target_by_prompt(base_dir, prompt):
    """ 根據 Prompt 尋找對應的 T2I 圖片 """
    if not prompt or not os.path.exists(base_dir): return None
    
    def normalize_quotes(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    
    target_normalized = normalize_quotes(prompt)
    for filename in os.listdir(base_dir):
        file_no_ext = os.path.splitext(filename)[0]
        if normalize_quotes(file_no_ext) == target_normalized:
            return os.path.join(base_dir, filename)
    return None

# ==========================================
# 1. 角度計算與模型載入
# ==========================================
def normalize_angle(angle):
    if angle is None: return 0.0 
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def limit_angle(angle):
    while angle < -180: angle += 360
    while angle > 180: angle -= 360
    return angle

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
        # Fallback manual calculation
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
# 2. 核心預測邏輯 (封裝版)
# ==========================================
def predict_pose_label(image_path, pose_detector, head_model, pose_classifier):
    """
    輸入圖片路徑，回傳預測的 Pose Label
    """
    if not image_path or not os.path.exists(image_path):
        return "Image_Not_Found"

    try:
        # 圖片處理
        img_pil = Image.open(image_path).convert("RGB")
        W, H = img_pil.size
        img_arr = np.array(img_pil)
        
        # 1. MediaPipe Pose
        results = pose_detector.process(img_arr)
        
        raw_body_yaw = None
        raw_body_roll = 0.0
        h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            raw_body_yaw = calc_body_yaw(lm)
            raw_body_roll = calc_body_roll(lm, W, H)
            
            # 2. Head Pose (如果檢測到人臉)
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
        
        # 3. 判斷是否有檢測到身體
        if raw_body_yaw is None:
            return "No_Body_Detected"
            
        # 4. 建構特徵向量 [BodyYaw, BodyPitch, BodyRoll, HeadYaw, HeadPitch, HeadRoll]
        norm_body = normalize_angle(raw_body_yaw)
        features = np.array([[
            norm_body,       # Body Yaw
            0.0,             # Body Pitch (MediaPipe 2D 不準，設為 0)
            raw_body_roll,   # Body Roll
            h_yaw,           # Head Yaw
            h_pitch,         # Head Pitch
            h_roll           # Head Roll
        ]])
        
        # 5. ML 預測
        prediction = pose_classifier.predict(features)[0]
        return prediction

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return "Error"

# ==========================================
# 3. JSON 處理主程式
# ==========================================
def run_pose_labeling(method, swapped_dir, t2i_dir, json_path, checkpoint_path, ml_model_path):
    if not HAS_MEDIAPIPE:
        print("❌ 錯誤: 未安裝 MediaPipe (pip install mediapipe)")
        return

    print(f"📂 JSON Path: {json_path}")
    print(f"📂 Swapped Dir: {swapped_dir}")
    print(f"📂 T2I Dir: {t2i_dir}")
    print(f"🧠 ML Model: {ml_model_path}")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # 1. 載入模型
    if os.path.exists(ml_model_path):
        try:
            pose_classifier = joblib.load(ml_model_path)
            print("✅ ML Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            return
    else:
        print(f"❌ ML model not found at {ml_model_path}")
        return

    mp_pose = mp_solutions.pose
    # 注意: 這裡不能用 with，因為我們要重複呼叫。static_image_mode=True
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    
    head_model = load_model_correctly(checkpoint_path)
    if head_model is None: return

    # 2. 讀取 JSON 資料
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Total items to process: {len(data_list)}")
    
    # 3. 處理迴圈
    for item in tqdm(data_list, desc="Pose Predicting (Swap & T2I)"):
        raw_filename = item.get("image", "").strip()
        prompt = item.get("prompt", "").strip()
        
        # --- A. 處理 Swapped Image ---
        swap_path, _ = smart_find_image(swapped_dir, raw_filename)
        pred_swap = predict_pose_label(swap_path, pose_detector, head_model, pose_classifier)
        item['pose_prediction'] = pred_swap
        
        # --- B. 處理 T2I Image ---
        t2i_path = find_target_by_prompt(t2i_dir, prompt)
        pred_t2i = predict_pose_label(t2i_path, pose_detector, head_model, pose_classifier)
        item['pose_prediction_t2i'] = pred_t2i

        # --- C. 自動比對正確性 (Ground Truth Check) ---
        gt_pose = item.get('gt_pose', '')
        
        # Swapped Correctness
        if gt_pose and gt_pose != 'Unknown':
            item['pose_correct'] = int(pred_swap == gt_pose)
        else:
            item['pose_correct'] = None
            
        # T2I Correctness (假設 GT 適用於原圖)
        if gt_pose and gt_pose != 'Unknown':
            item['pose_correct_t2i'] = int(pred_t2i == gt_pose)
        else:
            item['pose_correct_t2i'] = None

    # 4. 釋放資源
    pose_detector.close()

    # 5. 存檔
    print(f"\nSaving updated JSON to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print("✅ Process Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart', help="Mapping key for image directory")
    parser.add_argument("--json", type=str, default='gt.json', help="Input/Output JSON file")
    parser.add_argument("--checkpoint", default='./pose/checkpoints/SemiUHPE/DAD-WildHead-EffNetV2-S-best.pth', help="Head model path")
    parser.add_argument("--ml_model", default='./pose/pose_classifier_mediapipe.pkl', help="Path to the trained .pkl classifier")
    
    args = parser.parse_args()

    # 路徑映射
    path_map = {
        'pixart': './faceswap_results/pixart',
        'janus': './faceswap_results/janus',
        'infinity': './faceswap_results/infinity',
        'showo2': './faceswap_results/showo2'
    }
    
    # 定義資料夾
    # [FIX] 確保 t2i_dir 被正確定義
    t2i_dir = './pixart_outputs' 
    swapped_dir = path_map.get(args.method, './faceswap_results/pixart')

    # 執行主程式
    run_pose_labeling(args.method, swapped_dir, t2i_dir, args.json, args.checkpoint, args.ml_model)