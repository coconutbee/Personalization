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
    from src.fisher.fisher_utils import batch_torch_A_to_R
    HAS_REPO_UTILS = True
except ImportError:
    HAS_REPO_UTILS = False

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    import mediapipe.python.solutions as mp_solutions
    if hasattr(mp_solutions, 'pose'):
        HAS_MEDIAPIPE = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 關鍵點索引與頭部模型配置
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9 

# ==========================================
# 1. 核心預測邏輯 (與上一版相同)
# ==========================================
def find_target_by_prompt(base_dir, prompt):
    if not prompt or not os.path.exists(base_dir): return None
    def normalize_quotes(text):
        return text.replace("’", "'").replace("‘", "'").strip().lower()
    target_normalized = normalize_quotes(prompt)
    for filename in os.listdir(base_dir):
        file_no_ext = os.path.splitext(filename)[0]
        if normalize_quotes(file_no_ext) == target_normalized:
            return os.path.join(base_dir, filename)
    return None

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
        return pose_classifier.predict(features)[0]
    except: return "Error"

# ==========================================
# 2. 處理主程式 (追加結果到現有 JSON)
# ==========================================
def run_pose_update(json_file, swapped_dir, t2i_dir, checkpoint_path, ml_model_path):
    if not os.path.exists(json_file):
        print(f"❌ 找不到 JSON 檔案: {json_file}")
        return

    # 1. 載入模型
    try:
        pose_classifier = joblib.load(ml_model_path)
        head_model = load_head_model(checkpoint_path)
        pose_detector = mp_solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    except Exception as e:
        print(f"❌ 模型載入出錯: {e}")
        return

    # 2. 讀取現有的 JSON 資料
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    t2i_cache = {}
    
    # 用於統計正確率的計數器
    stats = {
        "swap_correct": 0,
        "t2i_correct": 0,
        "total": 0
    }

    print(f"🚀 開始預測 Pose 並與 GT 比對 (共 {len(data_list)} 筆資料)...")

    for item in tqdm(data_list):
        filename = item.get("image")
        prompt = item.get("prompt")
        gt_pose = item.get("gt_pose") # 從 JSON 讀取 Ground Truth
        
        if not filename or not prompt: continue
        
        swapped_path = os.path.join(swapped_dir, filename)
        t2i_path = find_target_by_prompt(t2i_dir, prompt)

        # --- A. 處理 Swapped Image ---
        pred_swap = predict_pose_label(swapped_path, pose_detector, head_model, pose_classifier)
        item["swap_pose_prediction"] = pred_swap
        
        # --- B. 處理 T2I Image (快取處理) ---
        if prompt not in t2i_cache:
            t2i_cache[prompt] = predict_pose_label(t2i_path, pose_detector, head_model, pose_classifier)
        pred_t2i = t2i_cache[prompt]
        item["t2i_pose_prediction"] = pred_t2i

        # --- C. 計算正確性 (Correctness) ---
        if gt_pose:
            # 比對預測與 GT (忽略大小寫與前後空白)
            is_swap_correct = 1 if str(pred_swap).strip().lower() == str(gt_pose).strip().lower() else 0
            is_t2i_correct = 1 if str(pred_t2i).strip().lower() == str(gt_pose).strip().lower() else 0
            
            item["swap_pose_correct"] = is_swap_correct
            item["t2i_pose_correct"] = is_t2i_correct
            
            # 統計
            stats["swap_correct"] += is_swap_correct
            stats["t2i_correct"] += is_t2i_correct
            stats["total"] += 1
        else:
            item["swap_pose_correct"] = None
            item["t2i_pose_correct"] = None

    # 3. 寫回 JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    
    pose_detector.close()

    # 4. 輸出統計結果
    if stats["total"] > 0:
        swap_acc = stats["swap_correct"] / stats["total"]
        t2i_acc = stats["t2i_correct"] / stats["total"]
        
        print("\n" + "="*50)
        print(f"📊 Pose Evaluation Results (Total: {stats['total']})")
        print("-" * 50)
        print(f"✅ Swapped Pose Accuracy: {swap_acc:.4f} ({stats['swap_correct']}/{stats['total']})")
        print(f"✅ T2I Pose Accuracy:     {t2i_acc:.4f} ({stats['t2i_correct']}/{stats['total']})")
        print("="*50)
    else:
        print("\n⚠️ 找不到 'gt_pose' 欄位，無法計算正確率。")

    print(f"✅ 檔案已更新完成至 {json_file}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="要追加結果的 JSON 檔案路徑")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--t2i", type=str, required=True)
    parser.add_argument("--swap", type=str, required=True)
    parser.add_argument("--checkpoint", default='./pose/checkpoints/SemiUHPE/DAD-WildHead-EffNetV2-S-best.pth')
    parser.add_argument("--ml_model", default='./pose/pose_classifier_mediapipe.pkl')
    
    args = parser.parse_args()

    run_pose_update(args.json, args.swap, args.t2i, args.checkpoint, args.ml_model)