import json
import re
import os

# ==========================================
# 1. 定義 Mapping Rules (Hard Coded)
# ==========================================

# --- Expression Mapping ---
EXPRESSION_MAPPING_RULES = {
    'happy': ['happy', 'grin', 'smile', 'smiles', 'grinning', 'smiling', 'laugh', 'giggles', 'joy', 'smirk', 'joyful', 'enjoy'],
    'surprise': ['surprised', 'surprise', 'amazed', 'astonished', 'skeptical', 'shock'],
    'confuse': ['confuse', 'forward', 'puzzled', 'questioning', 'thoughtful', 'confused'],
    'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene', 'calm'],
    'sad': ['sad', 'sadness', 'crying', 'gloomy', 'depressed'], 
    'others': ['others']
}

# --- Gender Mapping (Regex) ---
MALE_KEYWORDS = [r'\bman\b', r'\bboy\b', r'\bmale\b', r'\bmen\b', r'\bguy\b']
FEMALE_KEYWORDS = [r'\bwoman\b', r'\bgirl\b', r'\bfemale\b', r'\blady\b', r'\bwomen\b']

# --- Pose Mapping ---
# 為了確保長句子優先匹配 (例如 "looks up and to his left" 不會被 "looks up" 搶先)，
# 我們稍後會在函式中對這些 Key 依照長度排序。
POSE_MAPPING_RULES = {
    # --- Back View 類 ---
    "turns her head back over her shoulder": "Back_Over_Shoulder",
    "turns her head over her right shoulder": "Back_Over_Shoulder",

    # --- Head Turn 類 ---
    "turns her head left": "Head_Turn_Left",
    "looks sideways toward the left": "Head_Turn_Left",
    "turns his head right": "Head_Turn_Right",
    "looks to his right": "Head_Turn_Right",
    "turns his head slightly to the right": "Head_Slight_Right",

    # --- Tilt / Lean 類 ---
    "tilts his head left": "Head_Tilt_Left",
    "head tilted right": "Head_Tilt_Right",
    "leans his head toward his right shoulder": "Head_Tilt_Right",
    
    # --- Slight / Frontal 類 ---
    "looks straight": "Frontal",
    "tilts her head downward": "Frontal",
    "faces downward": "Frontal",
    "faces slightly downward": "Frontal",
    "looks down to her left": "Head_Slight_Left",
    "looks upward, head tilted back": "Frontal",
    "looks upward": "Frontal",
    "tilts her head backward": "Frontal",
    "looks up and to his left": "Head_Slight_Left",
    "turns his face upward to the left": "Head_Slight_Left",
}

# ==========================================
# 2. 定義標註邏輯函數
# ==========================================

def get_expression(text):
    text_lower = text.lower()
    for label, keywords in EXPRESSION_MAPPING_RULES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return label
    return "others"

def get_gender(text):
    text_lower = text.lower()
    is_male = False
    is_female = False
    
    for pattern in MALE_KEYWORDS:
        if re.search(pattern, text_lower):
            is_male = True
            break
            
    for pattern in FEMALE_KEYWORDS:
        if re.search(pattern, text_lower):
            is_female = True
            break
    
    if is_male and is_female: return "Both"
    elif is_male: return "Male"
    elif is_female: return "Female"
    else: return "Unknown"

def get_pose(text):
    text_lower = text.lower()
    
    # 關鍵步驟：依照字串長度排序 (由長到短)
    # 這樣可以避免 "looks up" 先匹配到 "looks up and to his left" 的情況
    sorted_keys = sorted(POSE_MAPPING_RULES.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        # 使用簡單的 substring check，只要 prompt 包含這個規則就當作匹配
        if key.lower() in text_lower:
            return POSE_MAPPING_RULES[key]
            
    return "Unknown"

# ==========================================
# 3. 主程式
# ==========================================

def process_json_data(input_data):
    """
    輸入: List of Dict (JSON 格式)
    輸出: 標註後的 List of Dict
    """
    labeled_data = []
    
    print(f"Processing {len(input_data)} items...")
    
    for item in input_data:
        prompt = item.get("prompt", "")
        
        # 進行標註
        item["gt_expression"] = get_expression(prompt)
        item["gt_gender"] = get_gender(prompt)
        item["gt_pose"] = get_pose(prompt)
        
        labeled_data.append(item)
        
    return labeled_data

if __name__ == "__main__":
    # 1. 設定檔案名稱
    input_filename = "prompts.json"       # 你的來源檔案
    output_filename = "gt.json"     # 處理後要存的檔案

    # 2. 讀取 JSON 檔案
    if os.path.exists(input_filename):
        try:
            print(f"Reading from {input_filename}...")
            with open(input_filename, 'r', encoding='utf-8') as f:
                # 這裡就是你要的：從檔案讀取並存入變數
                raw_json_input = json.load(f) 
                
            # 3. 執行處理
            result_data = process_json_data(raw_json_input)

            # 4. 儲存結果
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Processed {len(result_data)} items.")
            print(f"Results saved to: {output_filename}")

        except json.JSONDecodeError:
            print(f"Error: {input_filename} 的格式錯誤 (不是合法的 JSON)。")
    else:
        print(f"Error: 找不到檔案 {input_filename}，請確認檔案是否在同一個資料夾內。")