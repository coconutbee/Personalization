import streamlit as st
import json
import os
import pandas as pd
from PIL import Image

# ==========================================
# 1. 頁面與載入設定
# ==========================================
st.set_page_config(page_title="Pose Alignment Viewer", layout="wide")
st.title("🖼️ Pose Alignment 預測結果檢視器")

st.sidebar.header("⚙️ 設定與篩選")
json_path = st.sidebar.text_input("JSON 檔案路徑", "all_pose_angles.json")

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def extract_prompt_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if "_boy_" in name:
        target_prompt = name.split("_boy_")[-1]
    elif "_girl_" in name:
        target_prompt = name.split("_girl_")[-1]
    else:
        parts = name.split("_", 1)
        target_prompt = parts[1] if len(parts) > 1 else name
    return target_prompt.replace("_", " ")

# ==========================================
# 2. 核心計分與條件萃取邏輯 (獨立計算 平均得分)
# ==========================================
def parse_conditions(text):
    """將字串解析為 Yaw 與 Pitch 條件"""
    if not text:
        return None, None
    t = text.lower().replace("_", " ")
    
    yaw = None
    pitch = None
    
    if "forward" in t or "straight" in t:
        yaw = "center"
        pitch = "center" 
    elif "left over the shoulder" in t:
        yaw = "left_over"
    elif "right over the shoulder" in t:
        yaw = "right_over"
    elif "left" in t:
        yaw = "left"
    elif "right" in t:
        yaw = "right"
        
    if "tilted up" in t:
        pitch = "up"
    elif "tilted down" in t:
        pitch = "down"
        
    return yaw, pitch

def check_match_score(prompt_text, pred_text):
    """分別計算 Yaw 與 Pitch 的得分並取平均"""
    p_yaw, p_pitch = parse_conditions(prompt_text)
    r_yaw, r_pitch = parse_conditions(pred_text)
    
    if p_yaw is None and p_pitch is None:
        return None, p_yaw, p_pitch, r_yaw, r_pitch
        
    total_conditions = 0
    matched_conditions = 0
    
    if p_yaw is not None:
        total_conditions += 1
        if p_yaw == r_yaw:
            matched_conditions += 1
            
    if p_pitch is not None:
        total_conditions += 1
        if p_pitch == r_pitch:
            matched_conditions += 1
            
    score = matched_conditions / total_conditions
    return score, p_yaw, p_pitch, r_yaw, r_pitch

# ==========================================
# 3. 處理資料與呈現
# ==========================================
data = load_data(json_path)

if not data:
    st.warning(f"找不到或無法讀取 JSON 檔案：{json_path}。請確認路徑是否正確。")
else:
    total_score = 0.0
    valid_count = 0
    score_counts = {0.0: 0, 0.5: 0, 1.0: 0}
    
    # 預處理所有資料
    for item in data:
        gt_prompt = extract_prompt_from_filename(item.get("Filename", ""))
        pred_class = item.get("Prediction_Class", "Unknown")
        
        score, p_y, p_p, r_y, r_p = check_match_score(gt_prompt, pred_class)
        
        item["gt_prompt"] = gt_prompt
        item["score"] = score
        item["parsed_prompt"] = f"Yaw: `{p_y}`, Pitch: `{p_p}`"
        item["parsed_pred"] = f"Yaw: `{r_y}`, Pitch: `{r_p}`"
        
        if score is not None:
            total_score += score
            valid_count += 1
            if score in score_counts:
                score_counts[score] += 1

    # --- 儀表板區域 ---
    overall_accuracy = (total_score / valid_count) * 100 if valid_count > 0 else 0
    
    st.markdown("### 📊 整體預測統計")
    col_metric, col_chart = st.columns([1, 2])
    
    with col_metric:
        st.metric(
            label="🏆 整體預測準確率 (Average Score)", 
            value=f"{overall_accuracy:.1f}%", 
            delta=f"總得分 {total_score:.1f} / {valid_count} 張有效圖片"
        )
        st.info("**計分規則**：如果 Prompt 同時有 Yaw 與 Pitch 條件，各佔 0.5 分；單向條件佔 1.0 分。")

    with col_chart:
        # 將得分統計轉換為 pandas DataFrame 來繪製直方圖
        chart_data = pd.DataFrame(
            {"數量": [score_counts[0.0], score_counts[0.5], score_counts[1.0]]},
            index=["0.0 (完全錯誤)", "0.5 (部分相符)", "1.0 (完全相符)"]
        )
        st.bar_chart(chart_data)
        
    st.markdown("---")

    # --- 側邊欄過濾器 ---
    all_classes = sorted(list(set([item.get("Prediction_Class", "Unknown") for item in data])))
    selected_class = st.sidebar.selectbox("🔎 篩選預測類別", ["All"] + all_classes)
    match_filter = st.sidebar.radio("✅ 篩選配對得分", ["All", "完全相符 (Score 1.0)", "部分相符 (Score 0.5)", "完全錯誤 (Score 0.0)"])

    # 套用過濾
    filtered_data = [d for d in data if d.get("score") is not None]
    
    if selected_class != "All":
        filtered_data = [d for d in filtered_data if d.get("Prediction_Class") == selected_class]
        
    if match_filter == "完全相符 (Score 1.0)":
        filtered_data = [d for d in filtered_data if d.get("score") == 1.0]
    elif match_filter == "部分相符 (Score 0.5)":
        filtered_data = [d for d in filtered_data if d.get("score") == 0.5]
    elif match_filter == "完全錯誤 (Score 0.0)":
        filtered_data = [d for d in filtered_data if d.get("score") == 0.0]

    st.write(f"目前顯示 **{len(filtered_data)}** 筆資料")

    # --- 圖片矩陣顯示 ---
    cols_per_row = 3
    for i in range(0, len(filtered_data), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(filtered_data):
                item = filtered_data[i + j]
                with cols[j]:
                    img_path = item.get("Path", "")
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.error(f"找不到圖片：{img_path}")

                    score_val = item.get("score", 0)
                    if score_val == 1.0:
                        status_ui = "✅ **1.0 (完全相符)**"
                    elif score_val == 0.5:
                        status_ui = "⚠️ **0.5 (部分相符)**"
                    else:
                        status_ui = "❌ **0.0 (完全錯誤)**"
                        
                    st.markdown(f"**得分**: {status_ui}")
                    st.markdown(f"**🎯 Prompt:** `{item['gt_prompt']}`")
                    st.caption(f"提取條件 -> {item['parsed_prompt']}")
                    
                    if score_val == 1.0:
                        st.markdown(f"**🤖 預測:** `{item.get('Prediction_Class', 'Unknown')}`")
                    else:
                        st.markdown(f"**🤖 預測:** :red[{item.get('Prediction_Class', 'Unknown')}]")
                    st.caption(f"提取條件 -> {item['parsed_pred']}")

                    with st.expander("📐 查看角度數據"):
                        st.json(item.get("Angles", {}))
        st.markdown("---")