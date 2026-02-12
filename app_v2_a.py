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
    if not filename_hint or not base_dir or not os.path.exists(base_dir): return None, "Dir Not Found"
    candidates = [filename_hint]
    if "’" in filename_hint or "‘" in filename_hint:
        candidates.append(filename_hint.replace("’", "'").replace("‘", "'"))
    candidates.append(f"0_{filename_hint}")
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path): return full_path, "Found"
    return None, f"Tried: {candidates}"

def find_target_by_prompt(base_dir, prompt):
    if not prompt or not base_dir or not os.path.exists(base_dir): return None
    def normalize(text): return text.replace("’", "'").replace("‘", "'").strip().lower()
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. 直接匹配
        if normalize(name_no_ext) == target_norm:
            return os.path.join(base_dir, filename)
            
        # 2. 忽略 ID 前綴匹配 (例如 00051_prompt.jpg)
        if "_" in name_no_ext:
            parts = name_no_ext.split('_', 1)
            if len(parts) > 1 and normalize(parts[1]) == target_norm:
                return os.path.join(base_dir, filename)
                
    return None

# ==========================================
# 🔄 資料處理
# ==========================================
def normalize_fga(raw_score):
    if raw_score is None: return 0.0
    val = float(raw_score)
    return min(max((val - 1.0) / 4.0, 0.0), 1.0)

def normalize_cols(df):
    cols_to_fill = [
        't2i_scenario_score', 't2i_pose_correct', 't2i_id_similarity', 'expression_correct_t2i',
        't2i_clip_t2i', 't2i_clip_id_i2i', 't2i_dino_id_i2i',
        # Swap 相關欄位 (若不存在則補0，但不影響 T2I 分析)
        'swap_scenario_score', 'swap_pose_correct', 'swap_id_similarity', 'expression_correct',
        'swap_clip_t2i', 'swap_clip_id_i2i', 'swap_dino_id_i2i'
    ]
    
    text_cols = ['gt_expression', 't2i_vlm_expression', 'swap_vlm_expression',
                 'gt_pose', 't2i_pose_prediction', 'swap_pose_prediction']
                 
    for col in text_cols:
        if col not in df.columns: df[col] = "N/A"
        else: df[col] = df[col].fillna("N/A").astype(str)

    for col in cols_to_fill:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0).astype(float)
    return df

@st.cache_data
def load_data(json_path):
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty: return df
    
    # ID 處理 (相容字串與數字 ID)
    if 'id' in df.columns: df['rand_id'] = df['id'].astype(str)
    elif 'rand_id' in df.columns: df['rand_id'] = df['rand_id'].astype(str)
    else: df['rand_id'] = "Unknown"

    if 'image' in df.columns:
        df = df.drop_duplicates(subset=['image'], keep='last')

    df = normalize_cols(df)
    
    # 處理 EvalMuse FGA 分數
    if 'fga_orig_score' in df.columns:
        df['evalmuse_t2i'] = df['fga_orig_score'].apply(normalize_fga)
    else: df['evalmuse_t2i'] = 0.0
        
    if 'fga_swap_score' in df.columns:
        df['evalmuse_swap'] = df['fga_swap_score'].apply(normalize_fga)
    else: df['evalmuse_swap'] = 0.0
    
    # 計算 Delta (若 Swap 存在)
    df['delta_total'] = df['swap_final_score'] - df['t2i_final_score'] if 'swap_final_score' in df.columns else 0
    df['delta_id'] = df['swap_id_similarity'] - df['t2i_id_similarity']
    
    return df

# ==========================================
# 📊 繪圖與樣式
# ==========================================
def color_delta_cells(val):
    if pd.isna(val) or val == 0: return ''
    return 'background-color: #d4edda; color: #155724' if val > 0 else 'background-color: #f8d7da; color: #721c24'

def plot_score_distributions(df, has_swap=False):
    metrics = [
        ('t2i_clip_t2i', 'CLIP Text Align (T)'),
        ('t2i_clip_id_i2i', 'CLIP ID Sim (T)'),
        ('t2i_dino_id_i2i', 'DINO ID Sim (T)'),
        ('evalmuse_t2i', 'EvalMuse (T)'),
        ('t2i_final_score', 'Final Score (T)')
    ]
    
    if has_swap:
        metrics.extend([
            ('swap_final_score', 'Final Score (S)'),
            ('evalmuse_swap', 'EvalMuse (S)')
        ])
    
    # 動態調整欄位數
    cols = st.columns(min(len(metrics), 5))
    
    for i, (col, title) in enumerate(metrics):
        if col not in df.columns: continue
        
        # 使用 modulo 來循環放置欄位
        with cols[i % 5]:
            mu = df[col].mean()
            sigma = df[col].std()
            
            chart_data = df[[col]].copy()
            base = alt.Chart(chart_data).encode(x=alt.X(col, bin=alt.Bin(maxbins=20), title=None))
            hist = base.mark_bar(opacity=0.7).encode(y=alt.Y('count()', title=None))
            rule = base.mark_rule(color='red').encode(x=alt.datum(mu), size=alt.value(2))
            
            st.markdown(f"**{title}**")
            st.altair_chart((hist + rule).properties(height=120), use_container_width=True)
            st.caption(f"μ: `{mu:.4f}` | σ: `{sigma:.4f}`")

def plot_fga_heatmap(item, has_swap=False):
    tokens_orig = item.get('fga_orig_tokens', [])
    tokens_swap = item.get('fga_swap_tokens', [])
    
    dfs = []
    if isinstance(tokens_orig, list) and tokens_orig:
        df_t = pd.DataFrame(tokens_orig, columns=['Token', 'Score'])
        df_t['Type'] = '1. T2I Original'
        df_t['Idx'] = range(len(df_t))
        dfs.append(df_t)
        
    if has_swap and isinstance(tokens_swap, list) and tokens_swap:
        df_s = pd.DataFrame(tokens_swap, columns=['Token', 'Score'])
        df_s['Type'] = '2. Swapped Result'
        df_s['Idx'] = range(len(df_s))
        dfs.append(df_s)
        
    if not dfs: 
        st.info("ℹ️ No FGA token data available.")
        return

    df_all = pd.concat(dfs)
    df_all['Label'] = df_all.apply(lambda x: f"{x['Idx']}: {x['Token']}", axis=1)
    
    chart = alt.Chart(df_all).mark_bar().encode(
        x=alt.X('Score:Q', title="Attention Score"),
        y=alt.Y('Label:N', title=None, sort=alt.EncodingSortField(field='Idx', order='ascending')),
        color=alt.condition(alt.datum.Score < 1.5, alt.value('#ff4b4b'), alt.value('#00cc96')),
        tooltip=['Type', 'Token', 'Score']
    ).properties(width=300, height=max(400, int(len(df_all)/2 * 25))).facet(column=alt.Column('Type:N', title=None)).resolve_scale(y='shared', x='shared')
    
    st.altair_chart(chart, use_container_width=True)


def display_metric_summary(df):
    """
    計算並展示 T2I 五大指標的平均數與標準差
    """
    # 定義要比較的五個指標 (顯示名稱 : DataFrame欄位名)
    target_metrics = {
        'CLIP Text Align': 't2i_clip_t2i',
        'CLIP ID Sim': 't2i_clip_id_i2i',
        'DINO ID Sim': 't2i_dino_id_i2i',
        'EvalMuse Score': 'evalmuse_t2i',
        'Final Score': 't2i_final_score'
    }

    stats_data = []
    
    # 計算統計量
    for label, col in target_metrics.items():
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            stats_data.append({
                "Metric": label,
                "Mean": mu,
                "Std Dev (σ)": sigma,
                "Min": df[col].min(),
                "Max": df[col].max()
            })
    
    if not stats_data:
        st.warning("No metric data available to summarize.")
        return

    stats_df = pd.DataFrame(stats_data)

    # --- 顯示介面 ---
    st.subheader("📈 T2I Metrics Statistics")
    
    c1, c2 = st.columns([2, 3])
    
    with c1:
        # 1. 數據表格 (使用 Pandas Styler 上色)
        st.caption("Statistical Summary Table")
        st.dataframe(
            stats_df.style.format({
                "Mean": "{:.4f}", 
                "Std Dev (σ)": "{:.4f}",
                "Min": "{:.3f}",
                "Max": "{:.3f}"
            }).background_gradient(subset=['Mean'], cmap='Blues'),
            use_container_width=True,
            hide_index=True
        )

    with c2:
        # 2. 視覺化比較圖 (Bar Chart with Error Bars)
        # 為了畫圖，我們需要將數據轉換格式
        st.caption("Mean Score with Standard Deviation Range")
        
        base = alt.Chart(stats_df).encode(
            x=alt.X('Metric', sort=list(target_metrics.keys()), title=None),
            tooltip=['Metric', 'Mean', 'Std Dev (σ)']
        )

        # 長條圖 (平均值)
        bars = base.mark_bar(color='#4c78a8', opacity=0.8).encode(
            y=alt.Y('Mean', title='Score')
        )

        # 誤差線 (標準差)
        error_bars = base.mark_errorbar(extent='ci').encode(
            y=alt.Y('Mean', title=''),
            yError='Std Dev (σ)'
        )
        
        # 疊加圖表
        chart = (bars + error_bars).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

def analyze_t2i_components(df):
    """
    針對 Pose, Expression, ID, Scenario 四個細項進行 T2I 分析
    """
    # 1. 定義顯示名稱與對應的 DataFrame 欄位
    components = {
        'Pose': 't2i_pose_correct',
        'Expression': 'expression_correct_t2i', 
        'Identity': 't2i_id_similarity',
        'Scenario': 't2i_scenario_score'
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
        json_path = st.text_input("JSON Path", value="photomaker_pslz_metadata.json")
        method_options = {'photomaker': ('/media/ee303/disk2/style_generation/PhotoMaker/photomaker_pslz')}
        
        # 預設路徑
        T2I_DIR_DEF = '/media/ee303/disk2/style_generation/PhotoMaker/photomaker_pslz'
        
        T2I_DIR = st.text_input("T2I Directory", T2I_DIR_DEF)
        REF_DIR = st.text_input("Reference Directory", "/media/ee303/disk2/JACK/reference")
        
        # Swapped 為選填，若未填則進入 T2I Only 模式
        SWAP_DIR = st.text_input("Swap Directory (Optional)", "")
        
        st.divider()
        st.success(f"T2I Found") if os.path.exists(T2I_DIR) else st.error(f"Missing: {T2I_DIR}")
        st.success(f"Ref Found") if os.path.exists(REF_DIR) else st.error(f"Missing: {REF_DIR}")
        
        HAS_SWAP = False
        if SWAP_DIR and os.path.exists(SWAP_DIR):
            HAS_SWAP = True
            st.success(f"Swap Found (Full Mode)")
        else:
            st.info("Swap Not Found (T2I Only Mode)")

    df = load_data(json_path)
    if df is None or df.empty:
        st.error(f"❌ JSON file not found or empty: `{json_path}`")
        return

    st.title(f"📊 EvalMuse Analytics: {'Full Comparison' if HAS_SWAP else 'T2I Analysis'}")

    # 1. 總覽
    avg_t2i = df['t2i_final_score'].mean()
    
    cols = st.columns(4)
    cols[0].metric("EvalMuse Score (T2I)", f"{avg_t2i:.3f}")
    cols[1].metric("Total Images", len(df))
    
    if HAS_SWAP:
        avg_swap = df['swap_final_score'].mean()
        cols[2].metric("EvalMuse Score (Swap)", f"{avg_swap:.3f}", delta=f"{avg_swap - avg_t2i:+.3f} vs T2I")
    
    st.divider()
    # display_metric_summary(df)
    analyze_t2i_components(df)
    # 2. 詳細檢視
    st.header("2. Detailed Inspection")
    
    # 選擇檢視案例
    sort_col = 't2i_final_score'
    sel_idx = st.selectbox(
        "Inspect Case (Sorted by Score):", 
        df.sort_values(sort_col, ascending=False).index, 
        format_func=lambda x: f"[ID:{df.loc[x]['rand_id']}] Score: {df.loc[x][sort_col]:.3f} | {df.loc[x]['prompt'][:50]}..."
    )
    
    item = df.loc[sel_idx]
    
    # 圖片展示區
    cols_img = st.columns(3 if HAS_SWAP else 2)
    
    # Ref
    path_ref, _ = smart_find_image(REF_DIR, f"{item['rand_id']}.png")
    with cols_img[0]: 
        st.markdown("**Reference (GT)**")
        if path_ref: st.image(Image.open(path_ref))
        st.caption(f"Expr: {item.get('gt_expression', 'N/A')} | Pose: {item.get('gt_pose', 'N/A')}")

    # T2I
    path_t2i = find_target_by_prompt(T2I_DIR, item['prompt'])
    with cols_img[1]: 
        st.markdown("**T2I**")
        if path_t2i: st.image(Image.open(path_t2i))
        st.caption(f"Expr: {item.get('t2i_vlm_expression', 'N/A')} | Pose: {item.get('t2i_pose', 'N/A')}")

    # Swap (Optional)
    if HAS_SWAP:
        path_swap, _ = smart_find_image(SWAP_DIR, item['image'])
        with cols_img[2]: 
            st.markdown("**Swap**")
            if path_swap: st.image(Image.open(path_swap))
            st.caption(f"Expr: {item.get('swap_vlm_expression', 'N/A')} | Pose: {item.get('swap_pose_prediction', 'N/A')}")

    # 分數表格
    st.subheader("🧬 Score Breakdown")
    
    metrics_data = [
        ["ID Similarity", 0.32, item['t2i_id_similarity']],
        ["Pose Correctness", 0.34, item['t2i_pose_correct']],
        ["Expression", 0.17, item['expression_correct_t2i']],
        ["Scenario Score", 0.17, item['t2i_scenario_score']],
        ["----------", np.nan, np.nan],
        ["**Final Score**", 1.00, item['t2i_final_score']]
    ]
    
    cols_data = ["Metric", "Weight", "T2I"]
    
    if HAS_SWAP:
        # 插入 Swap 數據
        metrics_data[0].insert(3, item['swap_id_similarity'])
        metrics_data[1].insert(3, item['swap_pose_correct'])
        metrics_data[2].insert(3, item['expression_correct'])
        metrics_data[3].insert(3, item['swap_scenario_score'])
        metrics_data[4].insert(3, np.nan)
        metrics_data[5].insert(3, item['swap_final_score'])
        cols_data.append("Swap")

    m_df = pd.DataFrame(metrics_data, columns=cols_data)
    st.dataframe(m_df.style.format({'Weight': '{:.2f}', 'T2I': '{:.3f}', 'Swap': '{:.3f}'}, na_rep=''), use_container_width=True)
    
    # Scenario Reasoning
    with st.expander("📝 Scenario Reasoning (VLM Output)"):
        st.markdown(f"**T2I Reasoning:**\n{item.get('t2i_scenario_reasoning', 'N/A')}")
        if HAS_SWAP:
            st.divider()
            st.markdown(f"**Swap Reasoning:**\n{item.get('swap_scenario_reasoning', 'N/A')}")

    # FGA Heatmap
    st.subheader("🔥 FGA Attention Heatmap")
    plot_fga_heatmap(item, has_swap=HAS_SWAP)

    st.divider()

    # 4. 原始數據總表
    st.header("4. Raw Metric Explorer")
    plot_score_distributions(df, has_swap=HAS_SWAP)

    raw_cols = [
        'rand_id', 'prompt', 't2i_final_score', 
        't2i_clip_t2i', 't2i_clip_id_i2i', 't2i_dino_id_i2i', 'evalmuse_t2i'
    ]
    if HAS_SWAP:
        raw_cols.extend(['swap_final_score', 'swap_clip_t2i', 'swap_clip_id_i2i', 'evalmuse_swap'])
        
    st.dataframe(
        df[raw_cols].style.format("{:.4f}", subset=[c for c in raw_cols if 'score' in c or 'clip' in c or 'dino' in c or 'evalmuse' in c]),
        use_container_width=True, height=500
    )

if __name__ == "__main__":
    main()