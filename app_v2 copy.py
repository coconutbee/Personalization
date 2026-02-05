import streamlit as st
import json
import os
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image

# ==========================================
# ⚙️ 全局設定
# ==========================================
st.set_page_config(layout="wide", page_title="FaceSwap Score Analysis", page_icon="📉")

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; }
    .delta-pos { color: #28a745; font-weight: bold; }
    .delta-neg { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📂 工具函式
# ==========================================
def smart_find_image(base_dir, filename_hint):
    """強健的圖片搜尋功能"""
    if not filename_hint or not os.path.exists(base_dir): 
        return None, "Dir Not Found"
    
    candidates = []
    candidates.append(filename_hint)
    
    if "’" in filename_hint or "‘" in filename_hint:
        normalized = filename_hint.replace("’", "'").replace("‘", "'")
        candidates.append(normalized)
        
    candidates.append(f"0_{filename_hint}")

    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path, "Found"
            
    return None, f"Tried: {candidates}"

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 尋找 T2I 原圖"""
    if not prompt or not os.path.exists(base_dir): return None
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')): continue
        name_no_ext = os.path.splitext(filename)[0]
        clean_name = name_no_ext[2:] if name_no_ext.startswith("0_") else name_no_ext
        if normalize(clean_name) == target_norm:
            return os.path.join(base_dir, filename)
    return None

# ==========================================
# 🔄 分數計算邏輯
# ==========================================
def recalculate_score_from_row(row, mode='swap'):
    W = {'expression': 0.17, 'scenario': 0.17, 'pose': 0.34, 'id': 0.32}
    
    if mode == 'swap':
        s_exp = float(row.get('expression_correct', 0))
        s_scen = float(row.get('swap_scenario_score', row.get('scenario_score', 0)))
        s_pose = float(row.get('swap_pose_correct', row.get('pose_correct', 0)))
        s_id = float(row.get('swap_id_similarity', row.get('id_similarity', 0)))
    else:
        s_exp = float(row.get('expression_correct_t2i', 0))
        s_scen = float(row.get('t2i_scenario_score', 0))
        s_pose = float(row.get('t2i_pose_correct', 0))
        s_id = float(row.get('t2i_id_similarity', 0))

    total_score = (s_exp * W['expression']) + (s_scen * W['scenario']) + (s_pose * W['pose']) + (s_id * W['id'])
    return round(total_score, 4)

@st.cache_data
def load_data(json_path):
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    if 'rand_id' in df.columns:
        df['rand_id'] = df['rand_id'].astype(str)
        df['rand_id'] = df['rand_id'].apply(lambda x: x.zfill(5) if len(x) < 5 else x)

    if not df.empty:
        df['swap_final_score'] = df.apply(lambda row: recalculate_score_from_row(row, 'swap'), axis=1)
        df['t2i_final_score'] = df.apply(lambda row: recalculate_score_from_row(row, 't2i'), axis=1)
        # 計算 Delta
        df['score_delta'] = df['swap_final_score'] - df['t2i_final_score']
        
    return df

# ==========================================
# 📊 圖表函式
# ==========================================
def plot_fga_heatmap(item):
    tokens_orig = item.get('fga_orig_tokens', [])
    tokens_swap = item.get('fga_swap_tokens', [])
    if not isinstance(tokens_orig, list) or not isinstance(tokens_swap, list):
        st.info("ℹ️ No FGA token data available.")
        return

    dfs = []
    if tokens_orig:
        df_t = pd.DataFrame(tokens_orig, columns=['Token', 'Score'])
        df_t['Type'] = '1. T2I Original'
        df_t['Idx'] = range(len(df_t))
        dfs.append(df_t)
    if tokens_swap:
        df_s = pd.DataFrame(tokens_swap, columns=['Token', 'Score'])
        df_s['Type'] = '2. Swapped Result'
        df_s['Idx'] = range(len(df_s))
        dfs.append(df_s)

    if not dfs: return

    df_all = pd.concat(dfs)
    df_all['Label'] = df_all.apply(lambda x: f"{x['Idx']}: {x['Token']}", axis=1)
    
    chart = alt.Chart(df_all).mark_bar().encode(
        x=alt.X('Score:Q', title="Attention Score"),
        y=alt.Y('Label:N', title=None, sort=alt.EncodingSortField(field='Idx', order='ascending')),
        color=alt.condition(alt.datum.Score < 1.5, alt.value('#ff4b4b'), alt.value('#00cc96')),
        tooltip=['Type', 'Token', 'Score']
    ).properties(width=300, height=max(400, int(len(df_all)/2 * 25))).facet(column=alt.Column('Type:N', title=None)).resolve_scale(y='shared', x='shared')
    st.altair_chart(chart, use_container_width=True)

def plot_correlation(df, x_col, y_col, x_name, y_name):
    cols_needed = {x_col, y_col, 'swap_final_score', 'image', 'prompt'}
    valid_cols = [c for c in cols_needed if c in df.columns]
    chart_df = df[valid_cols].copy()
    
    chart = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X(x_col, title=x_name),
        y=alt.Y(y_col, title=y_name),
        color=alt.Color('swap_final_score', scale=alt.Scale(scheme='turbo'), title='Final Score'),
        tooltip=[c for c in valid_cols]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_delta_scatter(df):
    """繪製 T2I vs Swap 的 Delta 散佈圖"""
    chart_df = df[['t2i_final_score', 'swap_final_score', 'score_delta', 'image', 'prompt']].copy()
    
    # 建立對角線 (x=y)
    line = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='gray', strokeDash=[5, 5]).encode(
        x='x', y='y'
    )
    
    # 散佈點
    points = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X('t2i_final_score', title='T2I Original Score'),
        y=alt.Y('swap_final_score', title='Swap Result Score'),
        color=alt.condition(
            alt.datum.score_delta > 0,
            alt.value('#28a745'),  # Green for Improved
            alt.value('#dc3545')   # Red for Worsened
        ),
        tooltip=['image', 't2i_final_score', 'swap_final_score', 'score_delta']
    ).interactive()

    st.altair_chart((points + line), use_container_width=True)

# ==========================================
# 🖥️ 主程式
# ==========================================
def main():
    with st.sidebar:
        st.title("🎛️ Settings")
        json_path = st.text_input("JSON Path", value="metadata.json")
        
        method_options = {
            'pixart': ('/media/ee303/disk2/JACK/FACE_SWAPED_pixart_test', '/media/ee303/disk2/style_generation/diffusers/pixart_test'),
        }
        method = st.selectbox("Method", list(method_options.keys()))
        
        SWAP_DIR_DEF, T2I_DIR_DEF = method_options[method]
        SWAP_DIR = st.text_input("Swap Directory", SWAP_DIR_DEF)
        T2I_DIR = st.text_input("T2I Directory", T2I_DIR_DEF)
        REF_DIR = st.text_input("Reference Directory", "/media/ee303/disk2/JACK/reference")
        
        st.divider()
        st.markdown("### Path Check")
        if os.path.exists(SWAP_DIR): st.success(f"Swap Dir Found")
        else: st.error(f"Swap Dir Missing: {SWAP_DIR}")
        if os.path.exists(REF_DIR): st.success(f"Ref Dir Found")
        else: st.error(f"Ref Dir Missing: {REF_DIR}")

        st.subheader("Filter")
        target_score = st.radio("Standard Deviation Basis:", ["Swap Score", "T2I Score"], index=0)
        score_col = 'swap_final_score' if "Swap" in target_score else 't2i_final_score'

    df = load_data(json_path)
    if df is None:
        st.error(f"❌ JSON file not found: `{json_path}`")
        return

    st.title(f"📊 Evaluation: {method.capitalize()}")

    # 1. 總覽
    avg_swap = df['swap_final_score'].mean()
    avg_t2i = df['t2i_final_score'].mean()
    std_swap = df['swap_final_score'].std()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Swap Score", f"{avg_swap:.3f}", delta=f"{avg_swap - avg_t2i:.3f} vs T2I")
    c2.metric("Avg T2I Score", f"{avg_t2i:.3f}")
    c3.metric("Std Dev (σ)", f"{std_swap:.3f}")
    c4.metric("Total Images", len(df))

    st.divider()

    # 2. 異常檢測
    st.header(f"2. Low Quality Detection")
    mean_val = df[score_col].mean()
    std_val = df[score_col].std()
    th_1std = mean_val - std_val
    th_2std = mean_val - (2 * std_val)
    if th_2std < 0: th_2std = 0.0

    st.info(f"Thresholds: -1σ = {th_1std:.3f}, -2σ = {th_2std:.3f}")
    
    df_critical = df[df[score_col] <= th_2std].sort_values(score_col)
    df_poor = df[(df[score_col] > th_2std) & (df[score_col] <= th_1std)].sort_values(score_col)
    df_normal = df[df[score_col] > th_1std]

    tab1, tab2, tab3 = st.tabs([f"🔴 Critical (< {th_2std:.2f})", f"🟠 Poor", "🟢 Normal"])
    
    # 預設資料集 (給下拉選單用)
    view_df = df_critical 

    with tab1:
        if df_critical.empty: st.success("No critical failures.")
        else: st.dataframe(df_critical[['image', 'swap_final_score', 'scenario_reasoning']], use_container_width=True)
    with tab2:
        if df_poor.empty: st.write("No poor samples.")
        else: 
            st.dataframe(df_poor[['image', 'swap_final_score', 'scenario_reasoning']], use_container_width=True)
            if df_critical.empty: view_df = df_poor
    with tab3:
        st.write("Normal samples hidden.")

    st.divider()

    # ==========================================
    # 3. 新增：T2I vs Swap 變化分析 (Impact Analysis)
    # ==========================================
    st.header("3. Swap Impact Analysis (Delta)")
    st.markdown("Analyze cases where FaceSwap **improved** or **worsened** the overall quality compared to the original T2I generation.")

    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        plot_delta_scatter(df)
        st.caption("Points **above** the dotted line = Improved. **Below** = Worsened.")

    with col_stats:
        improved_count = len(df[df['score_delta'] > 0])
        worsened_count = len(df[df['score_delta'] < 0])
        avg_delta = df['score_delta'].mean()
        
        st.metric("Total Improved Cases", improved_count, delta="Good")
        st.metric("Total Worsened Cases", worsened_count, delta="Bad", delta_color="inverse")
        st.metric("Average Score Change", f"{avg_delta:.4f}")

    # Delta 詳細列表
    df_improved = df[df['score_delta'] > 0].sort_values('score_delta', ascending=False)
    df_worsened = df[df['score_delta'] < 0].sort_values('score_delta', ascending=True)

    t_imp, t_wor = st.tabs([f"📈 Top Improvements ({len(df_improved)})", f"📉 Top Degradations ({len(df_worsened)})"])
    
    with t_imp:
        st.dataframe(df_improved[['image', 't2i_final_score', 'swap_final_score', 'score_delta']], use_container_width=True)
    with t_wor:
        st.dataframe(df_worsened[['image', 't2i_final_score', 'swap_final_score', 'score_delta']], use_container_width=True)

    st.divider()

    # 4. 視覺化檢查
    st.header("4. Visual Inspection")
    
    # 讓使用者選擇要從哪個資料集選圖
    inspect_source = st.radio("Inspect Samples From:", ["Low Quality Filter (Section 2)", "Impact Analysis - Improved", "Impact Analysis - Worsened"], horizontal=True)
    
    if inspect_source == "Low Quality Filter (Section 2)":
        target_view_df = view_df # 來自 Section 2 的預設
    elif inspect_source == "Impact Analysis - Improved":
        target_view_df = df_improved
    else:
        target_view_df = df_worsened

    if target_view_df.empty:
        st.warning("No samples in this category.")
    else:
        options = target_view_df.apply(lambda x: f"Score: {x['swap_final_score']:.3f} (Δ: {x['score_delta']:.3f}) | {x['prompt'][:60]}...", axis=1)
        selected_idx = st.selectbox("Select Case:", options.index, format_func=lambda x: options[x])
        item = df.loc[selected_idx]

        c1, c2, c3 = st.columns(3)
        ref_filename = f"{item['rand_id']}.png"
        path_ref, msg_ref = smart_find_image(REF_DIR, ref_filename)
        path_t2i = find_target_by_prompt(T2I_DIR, item['prompt'])
        swap_filename = item['image']
        path_swap, msg_swap = smart_find_image(SWAP_DIR, swap_filename)

        with c1:
            st.markdown(f"**Reference ({item['rand_id']})**")
            if path_ref: st.image(Image.open(path_ref), use_container_width=True)
            else: st.error(f"Missing: `{ref_filename}`")
        with c2:
            st.markdown(f"**T2I Original ({item.get('t2i_final_score',0):.3f})**")
            if path_t2i: st.image(Image.open(path_t2i), use_container_width=True)
            else: st.warning("T2I Missing")
        with c3:
            st.markdown(f"**Swap Result ({item.get('swap_final_score',0):.3f})**")
            if path_swap: st.image(Image.open(path_swap), use_container_width=True)
            else: st.error(f"Swap Missing")

        # 詳細數據對比
        st.subheader("🧬 Score Breakdown: T2I vs Swap")
        
        m1, m2 = st.columns([1, 2])
        with m1:
            delta_val = item['score_delta']
            delta_color = "green" if delta_val > 0 else "red"
            st.markdown(f"### Δ Change: <span style='color:{delta_color}'>{delta_val:+.4f}</span>", unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Expression", "Scenario", "Pose", "ID Sim"],
                "Weight": [0.17, 0.17, 0.34, 0.32],
                "T2I": [
                    float(item.get('expression_correct_t2i', 0)),
                    float(item.get('t2i_scenario_score', 0)),
                    float(item.get('t2i_pose_correct', 0)),
                    float(item.get('t2i_id_similarity', 0))
                ],
                "Swap": [
                    float(item.get('expression_correct', 0)),
                    float(item.get('swap_scenario_score', item.get('scenario_score', 0))),
                    float(item.get('swap_pose_correct', item.get('pose_correct', 0))),
                    float(item.get('swap_id_similarity', item.get('id_similarity', 0)))
                ]
            })
            # 計算各指標的 Delta
            metrics_df['Diff'] = metrics_df['Swap'] - metrics_df['T2I']
            
            st.dataframe(metrics_df, hide_index=True, use_container_width=True, 
                         column_config={"Diff": st.column_config.NumberColumn(format="%+.2f")})

            with st.expander("📝 Reasoning Comparison", expanded=False):
                t1, t2 = st.tabs(["T2I", "Swap"])
                with t1: st.write(item.get('scenario_reasoning_t2i', 'N/A'))
                with t2: st.write(item.get('scenario_reasoning', 'N/A'))

        with m2:
            st.markdown("**FGA Attention Analysis**")
            plot_fga_heatmap(item)

    st.divider()
    st.header("5. Metric Correlation")
    c1, c2 = st.columns(2)
    with c1:
        if 'swap_id_similarity' in df.columns:
            plot_correlation(df, 'swap_id_similarity', 'swap_final_score', 'ID Sim', 'Final Score')
    with c2:
        if 'swap_clip_t2i' in df.columns:
            plot_correlation(df, 'swap_clip_t2i', 'swap_final_score', 'CLIP Score', 'Final Score')

if __name__ == "__main__":
    main()