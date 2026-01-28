import streamlit as st
import json
import os
import pandas as pd
import altair as alt
from PIL import Image
import argparse

# ==========================================
# ⚙️ 設定區
# ==========================================


WEIGHTS = {
    'expression': 0.17,
    'scenario':   0.17,
    'gender':     0.17,
    'pose':       0.17,
    'id':         0.32
}

st.set_page_config(layout="wide", page_title="Face Swap & EvalMuse Benchmark", page_icon="🏆")

# ==========================================
# 📂 工具函式
# ==========================================
def calculate_score(item):
    s_exp  = float(item.get('expression_correct', 0) or 0)
    s_scen = float(item.get('scenario_score', 0.0) or 0.0)
    s_gen  = float(item.get('gender_correct', 0) or 0)
    s_pose = float(item.get('pose_correct', 0) or 0)
    
    raw_id = item.get('id_similarity')
    s_id = float(raw_id) if raw_id is not None else 0.0
    s_id = max(0.0, min(s_id, 1.0))

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
    for item in data:
        item['final_score'] = calculate_score(item)
        # 讀取 EvalMuse 分數
        item['fga_alignment_score_val'] = float(item.get('fga_alignment_score', 0) or 0)
        # 讀取 CLIP 分數
        item['clip_t2i_val'] = float(item.get('clip_t2i_score', 0) or 0)
        item['clip_i2i_val'] = float(item.get('clip_i2i_score', 0) or 0)
    return data

def smart_find_image(base_dir, filename_hint):
    if not filename_hint: return None
    name_no_ext = os.path.splitext(filename_hint)[0]
    candidates = [f"0_{filename_hint}", f"0_{name_no_ext}.png", filename_hint, f"{name_no_ext}.png"]
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
    return None

def find_target_by_prompt(base_dir, prompt):
    if not prompt: return None
    def normalize_quotes(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_normalized = normalize_quotes(prompt)
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            file_no_ext = os.path.splitext(filename)[0]
            if normalize_quotes(file_no_ext) == target_normalized:
                return os.path.join(base_dir, filename)
    return None

def find_ref_fixed(base_dir):
    candidates = ["0.png", "0.jpg", "0.jpeg"]
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
    return None

# ==========================================
# 🖥️ 主程式
# ==========================================
def main():
    st.title("🏆 UI application for visualization")
    data_list = load_and_process_data()
    if not data_list: return

    with st.sidebar:
        st.header("📂 Case Selector")
        options = [f"ID {item['id']} (Score: {item['final_score']})" for item in data_list]
        selected_option = st.selectbox("Select Case:", options)
        idx = options.index(selected_option)
        item = data_list[idx]
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()

    st.subheader(f"🔍 Case Inspection: ID {item['id']}")
    
    gen_path = smart_find_image(SWAPPED_DIR, item.get('image', '').strip())
    ref_path = find_ref_fixed(REF_DIR) 
    target_path = find_target_by_prompt(TARGET_DIR, item.get('prompt'))

    col_img, col_metrics = st.columns([1.5, 1])

    with col_img:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Reference (Scene)**")
            if ref_path: st.image(Image.open(ref_path), use_container_width=True)
        with c2:
            st.markdown("**2. Target (Face Source)**")
            if target_path: st.image(Image.open(target_path), use_container_width=True)
        with c3:
            st.markdown("**3. Generated (Result)**")
            if gen_path: st.image(Image.open(gen_path), use_container_width=True)
        
        with st.expander("📝 View Prompt & Reasoning"):
            st.markdown(f"**Prompt:** {item.get('prompt')}")
            st.divider()
            st.text(item.get('scenario_reasoning', 'No reasoning'))

    with col_metrics:
        st.markdown("#### 🎯 Metrics Comparison")
        # 擴展為 5 欄以容納所有分數
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🏆 Our Scoring Module", f"{item['final_score']:.1f}")
        k2.metric("🎨 EvalMuse", f"{item['fga_alignment_score_val']:.2f}")
        k3.metric("📝 CLIP T2I", f"{item['clip_t2i_val']:.2f}")
        k4.metric("🖼️ CLIP I2I", f"{item['clip_i2i_val']:.2f}")
        k5.metric("🦖 DINO", f"{float(item.get('dino_score', 0))*100:.1f}")
        
        st.divider()
        
        if st.button("📊 Show EvalMuse Word-Level Alignment"):
            token_scores = item.get('all_token_scores', [])
            if token_scores:
                df_tokens = pd.DataFrame(token_scores, columns=['Token', 'Score'])
                st.write("**Fine-grained Score per Token:**")
                token_chart = alt.Chart(df_tokens).mark_bar().encode(
                    x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 2.0])),
                    y=alt.Y('Token:N', sort=None),
                    color=alt.condition(alt.datum.Score > 0.8, alt.value("#2ecc71"), alt.value("#e74c3c"))
                ).properties(height=300)
                st.altair_chart(token_chart, use_container_width=True)

    st.divider()
    col_tab1, col_tab2 = st.tabs(["🧩 Component Details", "📈 Overall Benchmark"])
    
    with col_tab1:
        comp_df = pd.DataFrame([
            {"Metric": "ID Similarity (32%)", "Value": str(item.get('id_similarity', 'N/A'))},
            {"Metric": "Expression (17%)", "Value": f"{item.get('expression_correct')} (GT: {item.get('gt_expression')})"},
            {"Metric": "Scenario (17%)", "Value": str(item.get('scenario_score'))},
            {"Metric": "Gender (17%)", "Value": f"{item.get('gender_correct')} (GT: {item.get('gt_gender')})"},
            {"Metric": "Pose (17%)", "Value": f"{item.get('pose_correct')} (GT: {item.get('gt_pose')})"},
            {"Metric": "CLIP Text-Image Score", "Value": f"{item['clip_t2i_val']:.3f}"},
            {"Metric": "CLIP Image-Image Score", "Value": f"{item['clip_i2i_val']:.3f}"}
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with col_tab2:
        df_all = pd.DataFrame(data_list)
        avg_data = [
            {"Metric": "🏆 Final Score", "Score": df_all['final_score'].mean()},
            {"Metric": "🎨 EvalMuse (x50)", "Score": df_all['fga_alignment_score_val'].mean() * 50},
            {"Metric": "📝 CLIP T2I (x100)", "Score": df_all['clip_t2i_val'].mean() * 100},
            {"Metric": "🖼️ CLIP I2I (x100)", "Score": df_all['clip_i2i_val'].mean() * 100},
            {"Metric": "🦖 DINO Score (x100)", "Score": df_all['dino_score'].fillna(0).astype(float).mean() * 100},
        ]
        st.altair_chart(alt.Chart(pd.DataFrame(avg_data)).mark_bar().encode(
            x='Score:Q', y=alt.Y('Metric:N', sort='-x'), color='Metric:N'
        ).properties(height=250), use_container_width=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='EM_results/fga_word_scores.json')
    parser.add_argument('--method', type=str, default='pixart')
    args = parser.parse_args()
    if args.method == 'pixart':
        SWAPPED_DIR = "./faceswap_results/pixart"
        TARGET_DIR = "./pixart_outputs" 
    elif args.method == 'infinity':
        SWAPPED_DIR = "./faceswap_results/infinity"
        TARGET_DIR = "./infinity_outputs" 
    elif args.method == 'janus':
        SWAPPED_DIR = "./faceswap_results/janus"
        TARGET_DIR = "./janus_outputs"
    elif args.method == 'showo2':
        SWAPPED_DIR = "./faceswap_results/showo2"
        TARGET_DIR = "./showo2_outputs"
    REF_DIR = "./faceswap_results/reference"
    main()