import argparse
import json
import os
import sys
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import joblib 
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ==========================================
# 0. 環境與路徑設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.networks import get_EfficientNet_V2
    HAS_REPO_UTILS = True
except ImportError:
    HAS_REPO_UTILS = False
    # print("Warning: src.networks not found. Ensure you are in the correct directory.")

# 檢查 MediaPipe
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    import mediapipe.python.solutions as mp_solutions
    if hasattr(mp_solutions, 'pose'):
        HAS_MEDIAPIPE = True
except ImportError:
    print("❌ Error: mediapipe not installed. Please run 'pip install mediapipe'")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 關鍵點索引
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9 

# ==========================================
# 1. 核心預測邏輯
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

def load_head_model(checkpoint_path):
    print(f"📂 正在載入頭部姿勢模型: {checkpoint_path}")
    try:
        config = SOTAConfig()
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

def predict_pose_label(image_path, pose_detector, head_model, pose_classifier):
    """
    輸入圖片路徑，回傳 Pose Label (例如: 'front', 'left', 'right' 等)
    """
    if not image_path or not os.path.exists(image_path):
        return "Not_Found"
    try:
        img_pil = Image.open(image_path).convert("RGB")
        W, H = img_pil.size
        img_arr = np.array(img_pil)
        results = pose_detector.process(img_arr)
        
        raw_body_yaw, raw_body_roll = None, 0.0
        h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 計算身體 Yaw
            l_sh, r_sh = lm[IDX_L_SHOULDER], lm[IDX_R_SHOULDER]
            if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                dx, dz = r_sh.x - l_sh.x, r_sh.z - l_sh.z
                raw_body_yaw = -math.degrees(math.atan2(dz, dx)) * 2.0
                raw_body_roll = math.degrees(math.atan2(l_sh.y*H - r_sh.y*H, l_sh.x*W - r_sh.x*W))
            
            # 臉部切圖與頭部 Pose
            x_coords = [lm[i].x * W for i in FACE_LANDMARKS_INDICES]
            y_coords = [lm[i].y * H for i in FACE_LANDMARKS_INDICES]
            if x_coords:
                min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
                box_s = max(max_x - min_x, max_y - min_y) * 1.5
                cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
                crop = img_pil.crop((max(0, cx-box_s/2), max(0, cy-box_s/2), min(W, cx+box_s/2), min(H, cy+box_s/2)))
                if crop.size[0] > 10:
                    tf = transforms.Compose([
                        transforms.Resize((224, 224)), transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    with torch.no_grad():
                        out = head_model(tf(crop).unsqueeze(0).to(DEVICE))
                        h_yaw, h_pitch, h_roll = compute_pose_output(out)
        
        if raw_body_yaw is None: return "No_Body"
        features = np.array([[normalize_angle(raw_body_yaw), 0.0, raw_body_roll, h_yaw, h_pitch, h_roll]])
        
        # 使用 sklearn 的分類器預測標籤
        return pose_classifier.predict(features)[0]
    except Exception as e:
        # print(f"Error predicting pose for {image_path}: {e}")
        return "Error"

# ==========================================
# 2. 搜尋工具
# ==========================================
def find_ref_image_by_id(ref_dir, ref_id_str):
    """根據 ID (如 '00051') 找尋 Reference 圖片"""
    if not os.path.exists(ref_dir): return None
    
    # 優先找完全匹配
    for ext in ['.jpg', '.png', '.jpeg', '.webp']:
        exact_path = os.path.join(ref_dir, f"{ref_id_str}{ext}")
        if os.path.exists(exact_path):
            return exact_path
            
    # 前綴匹配
    for filename in os.listdir(ref_dir):
        if filename.startswith(f"{ref_id_str}_") or filename.startswith(f"{ref_id_str}."):
             return os.path.join(ref_dir, filename)
    return None

# ==========================================
# 3. 主處理邏輯 (T2I vs Reference)
# ==========================================
def process_pose_evaluation(task_name, json_path, t2i_dir, ref_dir, checkpoint_path, ml_model_path):
    print(f"\n🔹 Processing Task (Pose Accuracy): [{task_name}]")

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 檔案: {json_path}")
        return

    # 1. 載入模型
    try:
        pose_classifier = joblib.load(ml_model_path)
        head_model = load_head_model(checkpoint_path)
        pose_detector = mp_solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    except Exception as e:
        print(f"❌ 模型載入出錯，請確認路徑: {e}")
        return

    # 2. 讀取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    print(f"📂 Loaded JSON with {len(data_list)} items.")

    # 3. 處理迴圈
    stats = {
        "correct": 0,
        "total": 0,
        "ref_not_found": 0,
        "t2i_not_found": 0
    }
    
    # Cache Reference Pose (因為同一個 ID 可能被重複使用)
    ref_pose_cache = {}

    print(f"🚀 Starting Pose Evaluation...")
    for item in tqdm(data_list, desc=f"   Running {task_name}"):
        img_filename = item.get("image") # e.g. "00051_cat.jpg"
        ref_id = str(item.get("id"))     # e.g. "00051"
        
        if not img_filename or not ref_id: continue

        # A. 取得 T2I 圖片路徑
        t2i_path = os.path.join(t2i_dir, img_filename)
        
        # B. 取得 Reference 圖片路徑
        ref_path = find_ref_image_by_id(ref_dir, ref_id)
        
        # C. 計算 Reference Pose (作為 GT)
        if not ref_path:
            stats["ref_not_found"] += 1
            item["ref_pose"] = "Ref_Missing"
            item["pose_match"] = 0
            continue
            
        if ref_id in ref_pose_cache:
            gt_pose_label = ref_pose_cache[ref_id]
        else:
            gt_pose_label = predict_pose_label(ref_path, pose_detector, head_model, pose_classifier)
            ref_pose_cache[ref_id] = gt_pose_label
            
        item["ref_pose"] = gt_pose_label

        # D. 計算 T2I Pose
        if not os.path.exists(t2i_path):
            stats["t2i_not_found"] += 1
            item["t2i_pose"] = "Img_Missing"
            item["pose_match"] = 0
            continue
            
        t2i_pose_label = predict_pose_label(t2i_path, pose_detector, head_model, pose_classifier)
        item["t2i_pose"] = t2i_pose_label

        # E. 比對 (計算 Accuracy)
        # 排除 Error 或 No_Body 的情況 (視需求而定，這裡嚴格比對)
        is_correct = 0
        if gt_pose_label not in ["Error", "Not_Found", "No_Body"] and \
           t2i_pose_label not in ["Error", "Not_Found", "No_Body"]:
            if str(gt_pose_label).lower() == str(t2i_pose_label).lower():
                is_correct = 1
        
        item["pose_match"] = is_correct
        
        if gt_pose_label not in ["Error", "Not_Found", "No_Body"]: # 只有當 Reference 有效時才計入分母
            stats["correct"] += is_correct
            stats["total"] += 1

    # 4. 寫回 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    
    pose_detector.close()

    # 5. 輸出統計
    accuracy = 0.0
    if stats["total"] > 0:
        accuracy = (stats["correct"] / stats["total"]) * 100
    
    return {
        "name": task_name,
        "accuracy": accuracy,
        "valid_samples": stats["total"],
        "ref_missing": stats["ref_not_found"],
        "t2i_missing": stats["t2i_not_found"]
    }

# ==========================================
# 4. 主程式入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2I Pose Evaluation against Reference")
    parser.add_argument("--json", type=str, required=True, help="Input/Output JSON file")
    parser.add_argument("--t2i", type=str, required=True, help="T2I Images folder")
    parser.add_argument("--ref", type=str, required=True, help="Reference Images folder")
    parser.add_argument("--name", type=str, default="pose_eval", help="Task Name")
    
    # 模型路徑 (請根據實際位置修改預設值)
    parser.add_argument("--checkpoint", default='./pose/checkpoints/SemiUHPE/DAD-WildHead-EffNetV2-S-best.pth', help="Head Pose Model Path")
    parser.add_argument("--ml_model", default='./pose/pose_classifier_mediapipe.pkl', help="ML Classifier Path")
    
    args = parser.parse_args()

    res = process_pose_evaluation(args.name, args.json, args.t2i, args.ref, args.checkpoint, args.ml_model)

    if res:
        print("\n" + "="*80)
        print(f"Task: {res['name']}")
        print(f"Pose Accuracy (T2I vs Ref): {res['accuracy']:.2f}%")
        print(f"Valid Comparisons: {res['valid_samples']}")
        if res['ref_missing'] > 0:
            print(f"⚠️ Reference Images Missing: {res['ref_missing']}")
        print("="*80)
        print(f"✅ Results updated in: {args.json}")