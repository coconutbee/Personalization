import argparse
import streamlit as st
import json
import os
import pandas as pd
import altair as alt
from PIL import Image

# ==========================================
# ⚙️ 設定區
# ==========================================
# 權重設定
WEIGHTS = {
    'expression': 0.17,
    'scenario':   0.17,
    'gender':     0.17,
    'pose':       0.17,
    'id':         0.32
}

st.set_page_config(layout="wide", page_title="Final Benchmark Dashboard", page_icon="🏆")

# ==========================================
# 📂 工具函式
# ==========================================
def calculate_score(item):
    """
    即時計算單張圖片的加權總分 (0-100)
    """
    s_exp  = float(item.get('expression_correct', 0) or 0)
    s_scen = float(item.get('scenario_score', 0.0) or 0.0)
    s_gen  = float(item.get('gender_correct', 0) or 0)
    s_pose = float(item.get('pose_correct', 0) or 0)
    
    # ID 分數處理 None
    raw_id = item.get('id_similarity')
    if raw_id is None:
        s_id = 0.0
    else:
        try:
            s_id = float(raw_id)
            s_id = max(0.0, min(s_id, 1.0))
        except:
            s_id = 0.0

    weighted_sum = (
        (s_exp  * WEIGHTS['expression']) +
        (s_scen * WEIGHTS['scenario']) +
        (s_gen  * WEIGHTS['gender']) +
        (s_pose * WEIGHTS['pose']) +
        (s_id   * WEIGHTS['id'])
    )
    return round(weighted_sum * 100, 2)

@st.cache_data
def load_and_process_data():
    if not os.path.exists(args.json):
        st.error(f"❌ 找不到檔案: {args.json}")
        return []
    
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 預先計算每一筆的 Final Score
    for item in data:
        item['final_score'] = calculate_score(item)
        
    return data

def smart_find_image(base_dir, filename_hint):
    if not filename_hint: return None
    name_no_ext = os.path.splitext(filename_hint)[0]
    candidates = [f"0_{filename_hint}", f"0_{name_no_ext}.png", filename_hint, f"{name_no_ext}.png"]
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
    return None

def smart_find_ref(base_dir, ref_id):
    candidates = [f"{ref_id}.jpg", f"{ref_id}.png", f"{ref_id}.jpeg", f"0_{ref_id}.jpg"]
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
    return None

# ==========================================
# 🖥️ 主程式
# ==========================================
def main():
    st.title("🏆 Face Swap Final Benchmark Dashboard")
    
    data_list = load_and_process_data()
    if not data_list: return

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("📂 Case Selector")
        # 顯示 ID 與 Final Score 預覽
        options = [f"ID {item['id']} (Score: {item['final_score']})" for item in data_list]
        selected_option = st.selectbox("Select Case:", options)
        
        # 找出 index
        idx = options.index(selected_option)
        item = data_list[idx]
        
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
            
        st.info(f"""
        **Weight Config:**
        - ID Sim: 32%
        - Expr: 17%
        - Scenario: 17%
        - Gender: 17%
        - Pose: 17%
        """)

    # ==========================================
    # 1. 上半部：個別圖片分析
    # ==========================================
    st.subheader(f"🔍 Case Inspection: ID {item['id']}")
    
    # 找圖
    raw_gen_name = item.get('image', '').strip()
    gen_path = smart_find_image(SWAPPED_DIR, raw_gen_name)
    try:
        ref_id = os.path.basename(gen_path).split('_')[0] if gen_path else "0"
    except: ref_id = "0"
    ref_path = smart_find_ref(REF_DIR, ref_id)

    col_img, col_metrics = st.columns([1, 1.5])

    # 左：圖片
    with col_img:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Reference**")
            if ref_path: st.image(Image.open(ref_path), use_container_width=True)
            else: st.warning("No Ref Image")
        with c2:
            st.markdown("**Generated**")
            if gen_path: st.image(Image.open(gen_path), use_container_width=True)
            else: st.warning("No Gen Image")
        
        with st.expander("📝 View Prompt & VLM Reasoning"):
            st.markdown(f"**Prompt:** {item.get('prompt')}")
            st.divider()
            st.text(item.get('scenario_reasoning', 'No reasoning'))

    # 右：分數指標
    with col_metrics:
        # 準備數據 (確保 None 轉為 0)
        s_final = item['final_score']
        s_dino = float(item.get('dino_score') or 0) * 100
        s_clip_i2i = float(item.get('clip_i2i_score') or 0) * 100
        s_clip_t2i = float(item.get('clip_t2i_score') or 0) * 100
        
        # 第一排：大指標
        st.markdown("#### 🎯 Core Metrics Comparison (0-100)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Final Score", f"{s_final:.1f}", help="Our Weighted Metric")
        k2.metric("🦖 DINO", f"{s_dino:.1f}", help="Structure Similarity")
        k3.metric("🖼️ CLIP I2I", f"{s_clip_i2i:.1f}", help="Semantic Similarity (Ref)")
        k4.metric("📝 CLIP T2I", f"{s_clip_t2i:.1f}", help="Prompt Alignment")
        
        st.divider()
        
        # 第二排：Final Score 組成細項
        st.markdown("#### 🧩 Final Score Components")
        
        # 製作 DataFrame 來顯示狀態
        comp_df = pd.DataFrame([
            {"Metric": "ID Similarity (32%)", "Value": item.get('id_similarity', 'N/A'), "Status": "-"},
            {"Metric": "Expression (17%)",    "Value": f"{item.get('expression_correct')}", "Status": f"GT: {item.get('gt_expression')} / Pred: {item.get('vlm_expression')}"},
            {"Metric": "Scenario (17%)",      "Value": f"{item.get('scenario_score')}",    "Status": "VLM Checked"},
            {"Metric": "Gender (17%)",        "Value": f"{item.get('gender_correct')}",    "Status": f"GT: {item.get('gt_gender')} / Pred: {item.get('mivolo_gender')}"},
            {"Metric": "Pose (17%)",          "Value": f"{item.get('pose_correct')}",      "Status": f"GT: {item.get('gt_pose')} / Pred: {item.get('pose_prediction')}"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ==========================================
    # 2. 下半部：全體平均比較圖表
    # ==========================================
    st.divider()
    st.subheader("📈 Overall Dataset Benchmarking (Average Scores)")
    
    # 計算平均值
    df_all = pd.DataFrame(data_list)
    
    avg_data = [
        {"Metric": "🏆 Final Score (Ours)", "Score": df_all['final_score'].mean()},
        {"Metric": "🦖 DINO (Ref-Gen)",     "Score": df_all['dino_score'].fillna(0).astype(float).mean() * 100},
        {"Metric": "🖼️ CLIP I2I (Ref-Gen)", "Score": df_all['clip_i2i_score'].fillna(0).astype(float).mean() * 100},
        {"Metric": "📝 CLIP T2I (Text-Gen)", "Score": df_all['clip_t2i_score'].fillna(0).astype(float).mean() * 100},
    ]
    df_chart = pd.DataFrame(avg_data)
    
    # 畫圖
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 100]), title="Average Score (0-100)"),
        y=alt.Y('Metric:N', sort='-x', title=None),
        color=alt.Color('Metric:N', legend=None),
        tooltip=['Metric', alt.Tooltip('Score', format='.1f')]
    ).properties(height=300)
    
    text = chart.mark_text(dx=3, align='left').encode(
        text=alt.Text('Score:Q', format='.1f')
    )
    
    st.altair_chart(chart + text, use_container_width=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='gt.json')
    parser.add_argument('--method', type=str, default='pixart')
    args = parser.parse_args()
    if args.method == 'pixart':
        SWAPPED_DIR = "./faceswap_results/pixart"
    elif args.method == 'infinity':
        SWAPPED_DIR = "./faceswap_results/infinity"
    elif args.method == 'janus':
        SWAPPED_DIR = "./faceswap_results/janus"
    elif args.method == 'showo2':
        SWAPPED_DIR = "./faceswap_results/showo2"
    REF_DIR = "./faceswap_results/reference"
    main()