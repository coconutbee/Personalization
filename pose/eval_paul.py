import os
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
import sys
import math
import mediapipe as mp
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 設定 ---
# 1. 模型權重路徑
CHECKPOINT_PATH = "checkpoints/SemiUHPE/DAD-WildHead-EffNetV2-S-best.pth"
# 2. 支援的圖片格式
SUPPORT_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 環境設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 嘗試載入依賴 (已徹底移除 pytorch3d 相關的 batch_torch_A_to_R)
try:
    from src.networks import get_EfficientNet_V2
    HAS_DEPS = True
    mp_pose = mp.solutions.pose
except ImportError as e:
    HAS_DEPS = False
    print(f"❌ 缺少必要套件 (src, mediapipe)，請檢查環境。詳細錯誤: {e}")

class SOTAConfig:
    def __init__(self):
        self.num_classes = 9

# --- 模型載入函數 ---
def load_models():
    if not HAS_DEPS: return None, None
    print(f"📂 Loading models to {DEVICE}...")
    try:
        # Load Head Model (SemiUHPE)
        config = SOTAConfig()
        head_model = get_EfficientNet_V2(config, model_name="S")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        state_dict = checkpoint.get('model_state_dict_ema', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        head_model.load_state_dict(new_state_dict, strict=True)
        head_model.to(DEVICE)
        head_model.eval()

        # Load MediaPipe
        pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        
        return head_model, pose_estimator
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, None

# --- 核心計算邏輯 ---
def limit_angle(angle):
    """將角度限制在 -180 到 180 之間，並保留正負號"""
    if angle is None: return 0.0
    angle = float(angle)
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def get_face_box(landmarks, w, h):
    xs = [landmarks[i].x * w for i in FACE_LANDMARKS]
    ys = [landmarks[i].y * h for i in FACE_LANDMARKS]
    if not xs: return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    size = max(x2-x1, y2-y1) * 1.5
    return [int(cx - size/2), int(cy - size/2), int(cx + size/2), int(cy + size/2)]

def calc_angles(lm, w, h):
    l = lm[IDX_L_SHOULDER]
    r = lm[IDX_R_SHOULDER]
    b_yaw = 0.0
    b_roll = 0.0
    
    if l.visibility > 0.5 and r.visibility > 0.5:
        dx = l.x - r.x
        dz = l.z - r.z
        b_yaw = limit_angle(-math.degrees(math.atan2(dz, dx)))
        
        lx_px, ly_px = l.x * w, l.y * h
        rx_px, ry_px = r.x * w, r.y * h
        dx_roll = rx_px - lx_px
        dy_roll = ry_px - ly_px
        b_roll = limit_angle(math.degrees(math.atan2(dy_roll, abs(dx_roll))))
        
    return b_yaw, b_roll

# --- 分類邏輯 (依照圖片規則 + >100 反轉機制) ---
def classify_pose(delta_yaw, head_pitch):
    if delta_yaw is None or head_pitch is None:
        return "Unknown"

    Y = delta_yaw
    P = head_pitch
    base_class = "Unknown"

    # 1. 依照表格初步分類
    if -30 <= Y <= 30:
        if -10 <= P <= 10: base_class = "head facing forward straight"
        elif P > 10: base_class = "head tilted up"
        elif P < -10: base_class = "head tilted down"
    elif 30 < Y <= 45:
        if -10 <= P <= 10: base_class = "head turned to his/her left"
        elif P > 10: base_class = "head turned to his/her left and tilted up"
        elif P < -10: base_class = "head turned to his/her left and tilted down"
    elif -45 <= Y < -30:
        if -10 <= P <= 10: base_class = "head turned to his/her right"
        elif P > 10: base_class = "head turned to his/her right and tilted up"
        elif P < -10: base_class = "head turned to his/her right and tilted down"
    elif Y > 45:
        if -10 <= P <= 10: base_class = "head turned to his/her left over the shoulder"
        elif P > 10: base_class = "head turned to his/her left over the shoulder and tilted up"
        elif P < -10: base_class = "head turned to his/her left over the shoulder and tilted down"
    elif Y < -45:
        if -10 <= P <= 10: base_class = "head turned to his/her right over the shoulder"
        elif P > 10: base_class = "head turned to his/her right over the shoulder and tilted up"
        elif P < -10: base_class = "head turned to his/her right over the shoulder and tilted down"
    else:
        base_class = "Error!!!!!!!"

    # 2. 如果 delta yaw 絕對值超過 100，左右預測相反
    if abs(Y) > 100:
        # 使用佔位符來做安全替換，避免 left 換成 right 後又立刻被換回 left
        base_class = base_class.replace("left", "{L}").replace("right", "{R}")
        base_class = base_class.replace("{L}", "right").replace("{R}", "left")

    return base_class

# --- 資料夾掃描函數 ---
def scan_images(folder_path):
    image_files = []
    print(f"🔍 正在掃描資料夾: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORT_EXT):
                image_files.append(os.path.join(root, file))
    print(f"✅ 找到 {len(image_files)} 張圖片")
    return image_files

# --- 主處理邏輯 ---
def process_folder(folder_path):
    head_model, pose_estimator = load_models()
    if not head_model: return

    image_paths = scan_images(folder_path)
    output_name = os.path.basename(folder_path)
    if not image_paths:
        print("❌ 資料夾內沒有圖片")
        return

    head_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    total_processed = 0

    print(f"🚀 開始提取所有圖片的角度資訊並進行分類...")

    for img_path in tqdm(image_paths):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            img_arr = np.array(pil_img)

            mp_results = pose_estimator.process(img_arr)
            b_yaw, b_roll = 0.0, 0.0
            h_yaw, h_pitch, h_roll = 0.0, 0.0, 0.0
            bbox = None
            
            if mp_results.pose_landmarks:
                lm = mp_results.pose_landmarks.landmark
                b_yaw, b_roll = calc_angles(lm, w, h)
                bbox = get_face_box(lm, w, h)

            if bbox:
                x1, y1, x2, y2 = bbox
                crop = pil_img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
                
                if crop.size[0] > 10 and crop.size[1] > 10:
                    with torch.no_grad():
                        input_t = head_tf(crop).unsqueeze(0).to(DEVICE)
                        out = head_model(input_t)
                        
                        # 使用 SVD 替換掉 pytorch3d
                        A = out.view(-1, 3, 3)
                        U, S, V = torch.linalg.svd(A)
                        R_mat = torch.matmul(U, V.transpose(1, 2))
                        if torch.det(R_mat) < 0:
                            V_fixed = V.clone()
                            V_fixed[:, :, 2] *= -1
                            R_mat = torch.matmul(U, V_fixed.transpose(1, 2))
                        rot_mat = R_mat.cpu().numpy()[0]
                        r = Rotation.from_matrix(np.transpose(rot_mat))
                        angles = r.as_euler("xyz", degrees=True)
                        
                        h_pitch = limit_angle(angles[0] - 180)
                        h_yaw = limit_angle(angles[1])
                        h_roll = limit_angle(angles[2])

            # 計算具有方向性 (正負號) 的 Delta Yaw
            delta_yaw = limit_angle(h_yaw - b_yaw)
            
            # 進行分類標籤預測
            pose_class = classify_pose(delta_yaw, h_pitch)

            results.append({
                "Filename": os.path.basename(img_path),
                "Path": img_path,
                "Prediction_Class": pose_class,
                "Angles": {
                    "BodyYaw": round(b_yaw, 1),
                    "BodyRoll": round(b_roll, 1),
                    "HeadYaw": round(h_yaw, 1),
                    "HeadPitch": round(h_pitch, 1),
                    "HeadRoll": round(h_roll, 1),
                    "DeltaYaw": round(delta_yaw, 1)
                }
            })
            total_processed += 1

        except Exception as e:
            pass

    print("\n" + "="*50)
    print(f"📊 執行完畢")
    print("="*50)
    print(f"成功提取角度的圖片數: {total_processed} / {len(image_paths)}")
    
    if results:
        # 將結果存成 JSON 檔案
        with open(f"{output_name}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 分類結果與所有角度數據已存至: {output_name}.json")
    else:
        print("\n⚠️ 沒有成功提取任何角度。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default='/media/ee303/disk2/style_generation/diffusers/pixart_pose_alignment_v4', help="圖片資料夾路徑")
    args = parser.parse_args()
    
    if os.path.exists(args.img_dir):
        process_folder(args.img_dir)
    else:
        print(f"❌ 錯誤: 找不到資料夾 {args.img_dir}")