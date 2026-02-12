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
st.set_page_config(layout="wide", page_title="EvalMuse Analysis", page_icon="📉")

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; }
    .pos-delta { background-color: #d4edda !important; color: #155724 !important; }
    .neg-delta { background-color: #f8d7da !important; color: #721c24 !important; }
    .label-box {
        padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; margin-top: 5px;
    }
    .label-correct { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .label-wrong { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .label-neutral { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
    
    /* 狀態顏色圖例說明 */
    .status-legend {
        display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; padding: 10px; background: #fff; border-radius: 5px; border: 1px solid #eee;
    }
    .status-dot { width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 5px; position: relative; top: 1px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📂 工具函式
# ==========================================
def smart_find_image(base_dir, filename_hint):
    if not filename_hint or not os.path.exists(base_dir): return None, "Dir Not Found"
    candidates = [filename_hint]
    if "’" in filename_hint or "‘" in filename_hint:
        candidates.append(filename_hint.replace("’", "'").replace("‘", "'"))
    candidates.append(f"0_{filename_hint}")
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path): return full_path, "Found"
    return None, f"Tried: {candidates}"

def find_target_by_prompt(base_dir, prompt):
    if not prompt or not os.path.exists(base_dir): return None
    def normalize(text): return text.replace("’", "'").replace("‘", "'").strip()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')): continue
        name_no_ext = os.path.splitext(filename)[0]
        clean_name = name_no_ext[2:] if name_no_ext.startswith("0_") else name_no_ext
        if normalize(clean_name) == target_norm: return os.path.join(base_dir, filename)
    return None

# ==========================================
# 🔄 資料處理
# ==========================================
def normalize_fga(raw_score):
    if raw_score is None: return 0.0
    val = float(raw_score)
    return min(max((val - 1.0) / 4.0, 0.0), 1.0)

def normalize_swap_cols(df):
    if 'swap_scenario_score' not in df.columns and 'scenario_score' in df.columns:
        df['swap_scenario_score'] = df['scenario_score']
    if 'swap_pose_correct' not in df.columns and 'pose_correct' in df.columns:
        df['swap_pose_correct'] = df['pose_correct']
    if 'swap_id_similarity' not in df.columns and 'id_similarity' in df.columns:
        df['swap_id_similarity'] = df['id_similarity']
    if 'expression_correct' not in df.columns:
         df['expression_correct'] = 0.0
         
    cols_to_fill = ['swap_scenario_score', 'swap_pose_correct', 'swap_id_similarity', 'expression_correct',
                    't2i_scenario_score', 't2i_pose_correct', 't2i_id_similarity', 'expression_correct_t2i',
                    'swap_clip_t2i', 't2i_clip_t2i', 'swap_clip_id_i2i', 't2i_clip_id_i2i',
                    'swap_dino_id_i2i', 't2i_dino_id_i2i']
    
    text_cols = ['gt_expression', 'swap_vlm_expression', 't2i_vlm_expression', 'swap_vlm_expression', 't2i_vlm_expression',
                 'gt_pose', 'swap_pose_prediction', 't2i_pose_prediction']
    for col in text_cols:
        if col not in df.columns: df[col] = "N/A"
        else: df[col] = df[col].fillna("N/A").astype(str)

    for col in cols_to_fill:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0).astype(float)
    return df

def recalculate_final_scores(df):
    W = {'expression': 0.17, 'scenario': 0.17, 'pose': 0.34, 'id': 0.32}
    df['swap_final_score'] = (
        df['expression_correct'] * W['expression'] +
        df['swap_scenario_score'] * W['scenario'] +
        df['swap_pose_correct'] * W['pose'] +
        df['swap_id_similarity'] * W['id']
    )
    df['t2i_final_score'] = (
        df['expression_correct_t2i'] * W['expression'] +
        df['t2i_scenario_score'] * W['scenario'] +
        df['t2i_pose_correct'] * W['pose'] +
        df['t2i_id_similarity'] * W['id']
    )
    return df

@st.cache_data
def load_data(json_path):
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty: return df
    
    if 'rand_id' in df.columns:
        df['rand_id'] = df['rand_id'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    else:
        df['rand_id'] = "Unknown"

    if 'image' in df.columns and 'rand_id' in df.columns:
        df = df.drop_duplicates(subset=['image', 'rand_id'], keep='last')

    df = normalize_swap_cols(df)
    df = recalculate_final_scores(df)
    
    if 'fga_swap_score' in df.columns:
        df['evalmuse_swap'] = df['fga_swap_score'].apply(normalize_fga)
    else:
        df['evalmuse_swap'] = 0.0
        
    if 'fga_orig_score' in df.columns:
        df['evalmuse_t2i'] = df['fga_orig_score'].apply(normalize_fga)
    else:
        df['evalmuse_t2i'] = 0.0
    
    df['delta_total'] = df['swap_final_score'] - df['t2i_final_score']
    df['delta_id'] = df['swap_id_similarity'] - df['t2i_id_similarity']
    df['delta_pose'] = df['swap_pose_correct'] - df['t2i_pose_correct']
    df['delta_expr'] = df['expression_correct'] - df['expression_correct_t2i']
    df['delta_scen'] = df['swap_scenario_score'] - df['t2i_scenario_score']

    return df

# ==========================================
# 📊 繪圖與樣式
# ==========================================
def color_delta_cells(val):
    if pd.isna(val) or val == 0: return ''
    return 'background-color: #d4edda; color: #155724' if val > 0 else 'background-color: #f8d7da; color: #721c24'

def plot_delta_scatter(df):
    chart_df = df[['t2i_final_score', 'swap_final_score', 'delta_total', 'image']].copy()
    line = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='gray', strokeDash=[5, 5]).encode(x='x', y='y')
    points = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X('t2i_final_score', title='T2I Original (EvalMuse)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('swap_final_score', title='Swap Result (EvalMuse)', scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(alt.datum.delta_total > 0, alt.value('#28a745'), alt.value('#dc3545')),
        tooltip=['image', 't2i_final_score', 'swap_final_score', alt.Tooltip('delta_total', format='+.3f')]
    ).interactive()
    st.altair_chart((points + line), use_container_width=True)

def plot_component_stats_chart(df):
    metrics = {'ID Similarity': 'delta_id', 'Pose': 'delta_pose', 'Expression': 'delta_expr', 'Scenario': 'delta_scen'}
    stats_data = []
    for label, col in metrics.items():
        improved = len(df[df[col] > 0.001])
        worsened = len(df[df[col] < -0.001])
        unchanged = len(df) - improved - worsened
        stats_data.append({'Metric': label, 'Status': 'Improved (🟢)', 'Count': improved, 'Order': 1})
        stats_data.append({'Metric': label, 'Status': 'Unchanged (⚪)', 'Count': unchanged, 'Order': 2})
        stats_data.append({'Metric': label, 'Status': 'Worsened (🔴)', 'Count': worsened, 'Order': 3})
    
    chart_df = pd.DataFrame(stats_data)
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('sum(Count)', stack="normalize", axis=alt.Axis(format='%'), title='Percentage'),
        y=alt.Y('Metric', title=None),
        color=alt.Color('Status', scale=alt.Scale(domain=['Improved (🟢)', 'Unchanged (⚪)', 'Worsened (🔴)'], range=['#28a745', '#e2e3e5', '#dc3545'])),
        order=alt.Order('Order', sort='ascending'),
        tooltip=['Metric', 'Status', 'Count', alt.Tooltip('sum(Count)', format='.1%', title='Percentage')]
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    return chart_df

def plot_score_distributions(df):
    """繪製 5 個關鍵指標的分佈圖，並顯示 Mean/Std"""
    metrics = [
        ('swap_clip_t2i', 'CLIP Text Align'),
        ('swap_clip_id_i2i', 'CLIP ID Sim'),
        ('swap_dino_id_i2i', 'DINO ID Sim'),
        ('evalmuse_swap', 'EvalMuse Score'),
        ('swap_final_score', 'Final Score')
    ]
    
    # 建立多欄位佈局
    cols = st.columns(len(metrics))
    
    for i, (col, title) in enumerate(metrics):
        if col not in df.columns: continue
        
        # 1. 計算該指標的 Mean 與 Std
        mu = df[col].mean()
        sigma = df[col].std()
        
        # 2. 準備繪圖資料 (只取該欄位以避免 PyArrow 錯誤)
        chart_data = df[[col]].copy()
        
        # 3. 設定圖表 (直方圖 + 平均線)
        base = alt.Chart(chart_data).encode(x=alt.X(col, bin=alt.Bin(maxbins=20), title=None))
        hist = base.mark_bar(opacity=0.7).encode(y=alt.Y('count()', title=None)) # 隱藏 Y 軸標題節省空間
        rule = base.mark_rule(color='red').encode(x=alt.datum(mu), size=alt.value(2))
        
        # 4. 在對應的欄位中顯示：標題 -> 圖表 -> 統計數據
        with cols[i]:
            st.markdown(f"**{title}**")
            st.altair_chart((hist + rule).properties(height=120), use_container_width=True)
            # 顯示統計數值
            st.caption(f"Mean ($\mu$): `{mu:.4f}`")
            st.caption(f"Std ($\sigma$): `{sigma:.4f}`")
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

def analyze_t2i_components(df):
    """
    針對 Pose, Expression, ID, Scenario 四個細項進行 T2I 分析
    """
    # 1. 定義顯示名稱與對應的 DataFrame 欄位
    components = {
        'Pose': 'swap_pose_correct',
        'Expression': 'expression_correct', 
        'Identity': 'swap_id_similarity',
        'Scenario': 'swap_scenario_score'
    }

    stats_data = []
    
    # 2. 計算統計數據
    for label, col in components.items():
        if col in df.columns:
            val_mean = df[col].mean()
            val_std = df[col].std()
            stats_data.append({
                "Component": label,
                "Mean": val_mean,
                "Std Dev": val_std,
                "Lower": max(0, val_mean - val_std), # 誤差線下限 (不小於0)
                "Upper": min(1, val_mean + val_std)  # 誤差線上限 (不大於1)
            })
    
    if not stats_data:
        st.warning("⚠️ Missing component data (Pose/Exp/ID/Scen).")
        return

    stats_df = pd.DataFrame(stats_data)

    # 3. 介面呈現
    st.subheader("🧩 Component Breakdown (T2I)")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        # 表格顯示
        st.caption("Score Statistics")
        # 修正重點：style.format 改用字典指定欄位，避免格式化到字串欄位 'Component'
        st.dataframe(
            stats_df[['Component', 'Mean', 'Std Dev']].style.format(
                {'Mean': '{:.4f}', 'Std Dev': '{:.4f}'}
            )
            .background_gradient(subset=['Mean'], cmap='Greens', vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True
        )

    with c2:
        # 圖表顯示 (Bar Chart + Error Bars)
        st.caption("Performance & Stability (Bar = Mean, Line = Std Dev)")
        
        # 基礎圖層
        base = alt.Chart(stats_df).encode(
            x=alt.X('Component', sort=['ID', 'Pose', 'Expression', 'Scenario'], title=None),
            tooltip=['Component', 'Mean', 'Std Dev']
        )

        # 長條圖 (平均分數)
        bars = base.mark_bar(color='#4c78a8', opacity=0.8).encode(
            y=alt.Y('Mean', title='Score (0-1)', scale=alt.Scale(domain=[0, 1]))
        )

        # 誤差線 (標準差範圍)
        error_bars = base.mark_rule(color='red', strokeWidth=2).encode(
            y='Lower',
            y2='Upper'
        )
        
        # 顯示數值標籤
        text = bars.mark_text(dy=-10, color='black').encode(
            text=alt.Text('Mean', format='.3f')
        )

        st.altair_chart((bars + error_bars + text).properties(height=300), use_container_width=True)
# ==========================================
# 🖥️ 主程式
# ==========================================
def main():
    with st.sidebar:
        st.title("🎛️ Settings")
        json_path = st.text_input("JSON Path", value="metadata.json")
        method_options = {'pixart': ('/media/ee303/disk2/JACK/FACE_SWAPED_pixart_test', '/media/ee303/disk2/style_generation/diffusers/pixart_test')}
        method = st.selectbox("Method", list(method_options.keys()))
        SWAP_DIR_DEF, T2I_DIR_DEF = method_options[method]
        SWAP_DIR = st.text_input("Swap Directory", SWAP_DIR_DEF)
        T2I_DIR = st.text_input("T2I Directory", T2I_DIR_DEF)
        REF_DIR = st.text_input("Reference Directory", "/media/ee303/disk2/JACK/reference")
        st.divider()
        st.success(f"Swap Found") if os.path.exists(SWAP_DIR) else st.error(f"Missing: {SWAP_DIR}")
        st.success(f"Ref Found") if os.path.exists(REF_DIR) else st.error(f"Missing: {REF_DIR}")

    df = load_data(json_path)
    if df is None or df.empty:
        st.error(f"❌ JSON file not found or empty: `{json_path}`")
        return

    st.title(f"📊 EvalMuse Analytics: {method.capitalize()}")

    # 1. 總覽
    avg_swap = df['swap_final_score'].mean()
    avg_t2i = df['t2i_final_score'].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EvalMuse Score (Swap)", f"{avg_swap:.3f}", delta=f"{avg_swap - avg_t2i:+.3f} vs T2I")
    c2.metric("EvalMuse Score (T2I)", f"{avg_t2i:.3f}")
    c3.metric("Std Dev (Swap)", f"{df['swap_final_score'].std():.3f}")
    c4.metric("Total Images", len(df))
    st.divider()
    analyze_t2i_components(df)

    # 2. 細項統計
    st.header("2. Component Impact Statistics & Inspection")
    col_viz, col_table = st.columns([2, 1])
    with col_viz: stats_df = plot_component_stats_chart(df)
    with col_table:
        pivot_stats = stats_df.pivot(index='Metric', columns='Status', values='Count').fillna(0).astype(int)
        st.dataframe(pivot_stats.div(pivot_stats.sum(axis=1), axis=0).mul(100).round(1).style.format("{:.1f}%"), use_container_width=True)

    st.markdown("#### 🔍 Drill Down: Detailed Visual Check")
    dd1, dd2 = st.columns(2)
    with dd1: sel_metric = st.selectbox("Metric", ["ID Similarity", "Pose", "Expression", "Scenario"])
    with dd2: sel_status = st.selectbox("Status", ["Improved (🟢)", "Worsened (🔴)", "Unchanged (⚪)"])
    
    metric_map = {"ID Similarity": "delta_id", "Pose": "delta_pose", "Expression": "delta_expr", "Scenario": "delta_scen"}
    t_col = metric_map[sel_metric]
    
    if "Improved" in sel_status: f_subset = df[df[t_col] > 0.001].sort_values(t_col, ascending=False)
    elif "Worsened" in sel_status: f_subset = df[df[t_col] < -0.001].sort_values(t_col, ascending=True)
    else: f_subset = df[(df[t_col] >= -0.001) & (df[t_col] <= 0.001)]
    
    st.success(f"Found **{len(f_subset)}** cases.")
    
    if sel_metric == "Scenario" and not f_subset.empty:
        st.markdown("##### 📊 Scenario Score Change Distribution")
        s_counts = f_subset[t_col].round(2).value_counts()
        is_imp = "Improved" in sel_status
        s_counts = s_counts.sort_index(ascending=not is_imp)
        dist_df = pd.DataFrame({"Score Change (Δ)": s_counts.index, "Count": s_counts.values})
        dist_df["Score Change (Δ)"] = dist_df["Score Change (Δ)"].apply(lambda x: f"{x:+.2f}")
        c_chart = alt.Chart(dist_df).mark_bar().encode(
            x=alt.X('Score Change (Δ)', sort=None), y='Count',
            color=alt.value('#28a745' if is_imp else '#dc3545'), tooltip=['Score Change (Δ)', 'Count']
        ).properties(height=180)
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1: st.altair_chart(c_chart, use_container_width=True)
        with c_col2: st.dataframe(dist_df, hide_index=True, use_container_width=True)

    if not f_subset.empty:
        sel_idx = st.selectbox("Inspect Case:", f_subset.index, format_func=lambda x: f"[ID:{f_subset.loc[x]['rand_id']}] Δ: {f_subset.loc[x][t_col]:+.3f} | {f_subset.loc[x]['image']}")
        item = df.loc[sel_idx]
        
        c1, c2, c3 = st.columns(3)
        path_ref, _ = smart_find_image(REF_DIR, f"{item['rand_id']}.png")
        path_t2i = find_target_by_prompt(T2I_DIR, item['prompt'])
        path_swap, _ = smart_find_image(SWAP_DIR, item['image'])
        
        gt_expr = item.get('gt_expression', 'N/A')
        t2i_expr = item.get('t2i_vlm_expression', item.get('t2i_vlm_expression', 'N/A'))
        swap_expr = item.get('swap_vlm_expression', item.get('swap_vlm_expression', 'N/A'))
        gt_pose = item.get('gt_pose', 'N/A')
        t2i_pose = item.get('t2i_pose_prediction', 'N/A')
        swap_pose = item.get('swap_pose_prediction', 'N/A')

        def render_label(label, gt, is_ref=False):
            if is_ref: return f'<div class="label-box label-neutral">GT(from prompt): {label}</div>'
            status = "label-correct" if label.lower() == gt.lower() else "label-wrong"
            return f'<div class="label-box {status}">{label}</div>'

        with c1: 
            st.markdown("**Reference (GT)**")
            if path_ref: st.image(Image.open(path_ref))
            if sel_metric == "Expression": st.markdown(render_label(gt_expr, gt_expr, True), unsafe_allow_html=True)
            elif sel_metric == "Pose": st.markdown(render_label(gt_pose, gt_pose, True), unsafe_allow_html=True)

        with c2: 
            st.markdown("**T2I**")
            if path_t2i: st.image(Image.open(path_t2i))
            if sel_metric == "Expression": st.markdown(render_label(t2i_expr, gt_expr), unsafe_allow_html=True)
            elif sel_metric == "Pose": st.markdown(render_label(t2i_pose, gt_pose), unsafe_allow_html=True)
            elif sel_metric == "ID Similarity": st.metric("ID Sim", f"{item['t2i_id_similarity']:.2f}")
            elif sel_metric == "Scenario": st.metric("Score", f"{item['t2i_scenario_score']:.2f}")

        with c3: 
            st.markdown("**Swap**")
            if path_swap: st.image(Image.open(path_swap))
            if sel_metric == "Expression": st.markdown(render_label(swap_expr, gt_expr), unsafe_allow_html=True)
            elif sel_metric == "Pose": st.markdown(render_label(swap_pose, gt_pose), unsafe_allow_html=True)
            elif sel_metric == "ID Similarity": st.metric("ID Sim", f"{item['swap_id_similarity']:.2f}", delta=f"{item['delta_id']:.2f}")
            elif sel_metric == "Scenario": st.metric("Score", f"{item['swap_scenario_score']:.2f}", delta=f"{item['delta_scen']:.2f}")

        st.subheader("🧬 EvalMuse Score Breakdown")
        metrics_data = [
            ["ID Similarity", 0.32, item['t2i_id_similarity'], item['swap_id_similarity'], item['delta_id']],
            ["Pose Correctness", 0.34, item['t2i_pose_correct'], item['swap_pose_correct'], item['delta_pose']],
            ["Expression", 0.17, item['expression_correct_t2i'], item['expression_correct'], item['delta_expr']],
            ["Scenario Score", 0.17, item['t2i_scenario_score'], item['swap_scenario_score'], item['delta_scen']],
            ["----------", np.nan, np.nan, np.nan, np.nan],
            ["**Final Score**", 1.00, item['t2i_final_score'], item['swap_final_score'], item['delta_total']]
        ]
        m_df = pd.DataFrame(metrics_data, columns=["Metric", "Weight", "T2I", "Swap", "Δ Diff"])
        st.dataframe(m_df.style.format({'Weight': '{:.2f}', 'T2I': '{:.3f}', 'Swap': '{:.3f}', 'Δ Diff': '{:+.3f}'}, na_rep='').applymap(color_delta_cells, subset=['Δ Diff']), use_container_width=True)
        
        with st.expander("📝 Reasoning"):
            t1, t2 = st.tabs(["T2I", "Swap"])
            with t1: st.write(item.get('t2i_scenario_reasoning', 'N/A'))
            with t2: st.write(item.get('swap_scenario_reasoning', 'N/A'))
        plot_fga_heatmap(item)

    st.divider()

    # 3. 進階過濾
    st.header("3. Advanced Filter")
    f1, f2, f3, f4 = st.columns(4)
    with f1: id_f = st.slider("Δ ID", -1.0, 1.0, (-0.1, 1.0), 0.05)
    with f2: po_f = st.slider("Δ Pose", -1.0, 1.0, (-0.1, 1.0), 0.05)
    with f3: ex_f = st.slider("Δ Expr", -1.0, 1.0, (-0.1, 1.0), 0.05)
    with f4: sc_f = st.slider("Δ Scen", -1.0, 1.0, (-1.0, 1.0), 0.05)
    
    adv_res = df[(df['delta_id'].between(id_f[0], id_f[1])) & (df['delta_pose'].between(po_f[0], po_f[1])) & 
                 (df['delta_expr'].between(ex_f[0], ex_f[1])) & (df['delta_scen'].between(sc_f[0], sc_f[1]))]
    st.write(f"Matches: **{len(adv_res)}**")
    if not adv_res.empty:
        idx = st.selectbox("Inspect:", adv_res.index, format_func=lambda x: f"[ID:{adv_res.loc[x]['rand_id']}] ΔTotal:{adv_res.loc[x]['delta_total']:+.2f} | {adv_res.loc[x]['image']}")
        st.write(df.loc[idx][['swap_final_score', 'delta_total', 'prompt']])

    st.divider()

    # ==========================================
    # 4. 原始分數總表與檢查 (含分佈圖與 Z-Score 著色)
    # ==========================================
    st.header("4. Raw Metric Explorer (Full Data Table)")
    
    st.markdown("##### 📈 Metric Distributions")
    plot_score_distributions(df)
    st.caption("Red line indicates the Mean.")

    st.markdown("##### 📋 Data Table (Colored by Z-Score)")
    st.markdown("""
    <div class="status-legend">
        <div><span class="status-dot" style="background-color: #ffcccc;"></span>< -2σ (Critical)</div>
        <div><span class="status-dot" style="background-color: #ffe5cc;"></span>< -1σ (Warning)</div>
        <div><span class="status-dot" style="background-color: #d4edda;"></span>< Mean (Below Avg)</div>
    </div>
    """, unsafe_allow_html=True)

    raw_cols = [
        'image', 'rand_id', 
        'swap_final_score', 't2i_final_score',
        'evalmuse_swap', 'evalmuse_t2i',
        'swap_clip_t2i', 't2i_clip_t2i', 
        'swap_clip_id_i2i', 't2i_clip_id_i2i',
        'swap_dino_id_i2i', 't2i_dino_id_i2i'
    ]
    raw_df = df[raw_cols].copy()
    
# --- 1. 修改 Z-Score 著色邏輯 (加入 EvalMuse) ---
    def style_raw_table(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        # 定義成對的指標群組 (加入 evalmuse_swap 與 evalmuse_t2i)
        metric_pairs = [
            ['swap_final_score', 't2i_final_score'],
            ['swap_clip_t2i', 't2i_clip_t2i'],
            ['swap_clip_id_i2i', 't2i_clip_id_i2i'],
            ['swap_dino_id_i2i', 't2i_dino_id_i2i'],
            ['evalmuse_swap', 'evalmuse_t2i']  # <--- 新增這一行
        ]
        
        for cols in metric_pairs:
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols: continue
            
            # 合併數據計算統一的 Mean 與 Std
            pooled_data = pd.concat([df[c] for c in valid_cols])
            mu = pooled_data.mean()
            sigma = pooled_data.std()
            
            def get_color(val):
                if pd.isna(val): return ''
                if val < mu - 2 * sigma: return 'background-color: #ffcccc; color: #990000' # Red
                if val < mu - 1 * sigma: return 'background-color: #ffe5cc; color: #cc6600' # Orange
                if val < mu:             return 'background-color: #d4edda; color: #155724' # Green
                return '' 
            
            for col in valid_cols:
                styles[col] = df[col].apply(get_color)
                
        return styles

    # 應用樣式
    st.dataframe(
        raw_df.style.apply(lambda _: style_raw_table(raw_df), axis=None)
                    .format("{:.4f}", subset=[c for c in raw_cols if c not in ['image', 'rand_id']]),
        use_container_width=True, 
        height=500,
        column_config={
            "swap_final_score": st.column_config.NumberColumn("Final (Swap)"),
            "t2i_final_score": st.column_config.NumberColumn("Final (T2I)"),
            "swap_clip_t2i": st.column_config.NumberColumn("CLIP-Txt (S)"),
            "t2i_clip_t2i": st.column_config.NumberColumn("CLIP-Txt (T)"),
            "swap_clip_id_i2i": st.column_config.NumberColumn("CLIP-ID (S)"),
            "t2i_clip_id_i2i": st.column_config.NumberColumn("CLIP-ID (T)"),
            "swap_dino_id_i2i": st.column_config.NumberColumn("DINO-ID (S)"),
            "t2i_dino_id_i2i": st.column_config.NumberColumn("DINO-ID (T)"),
        }
    )

    st.markdown("#### 🖼️ Row Visualizer")
    viz_options = df.apply(lambda x: f"[ID: {x['rand_id']}] Final: {x['swap_final_score']:.3f} | {x['image']}", axis=1)
    viz_idx = st.selectbox("Select Row to Visualize:", viz_options.index, format_func=lambda x: viz_options[x])
    
    if viz_idx is not None:
        v_item = df.loc[viz_idx]
        
        vc1, vc2, vc3 = st.columns(3)
        v_ref, _ = smart_find_image(REF_DIR, f"{v_item['rand_id']}.png")
        v_t2i = find_target_by_prompt(T2I_DIR, v_item['prompt'])
        v_swap, _ = smart_find_image(SWAP_DIR, v_item['image'])
        
        with vc1: 
            st.caption(f"Ref: {v_item['rand_id']}")
            if v_ref: st.image(Image.open(v_ref), use_container_width=True)
        with vc2: 
            st.caption("T2I Original")
            if v_t2i: st.image(Image.open(v_t2i), use_container_width=True)
        with vc3: 
            st.caption("Swapped Result")
            if v_swap: st.image(Image.open(v_swap), use_container_width=True)

        st.markdown("##### 📊 Raw Score Comparison")
        comp_data = {
            "Metric": ["CLIP Text Align", "CLIP ID Sim", "DINO ID Sim", "EvalMuse Score", "Final Score"],
            "T2I": [v_item.get('t2i_clip_t2i', 0), v_item.get('t2i_clip_id_i2i', 0), v_item.get('t2i_dino_id_i2i', 0), v_item.get('evalmuse_t2i', 0), v_item.get('t2i_final_score', 0)],
            "Swap": [v_item.get('swap_clip_t2i', 0), v_item.get('swap_clip_id_i2i', 0), v_item.get('swap_dino_id_i2i', 0), v_item.get('evalmuse_swap', 0), v_item.get('swap_final_score', 0)],
        }
        comp_df = pd.DataFrame(comp_data)
        comp_df["Delta"] = comp_df["Swap"] - comp_df["T2I"]

        st.dataframe(
            comp_df.style.format({"T2I": "{:.4f}", "Swap": "{:.4f}", "Delta": "{:+.4f}"})
            .applymap(color_delta_cells, subset=['Delta']),
            use_container_width=True, hide_index=True
        )

if __name__ == "__main__":
    main()