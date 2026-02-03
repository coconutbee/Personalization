import streamlit as st
import json
import os
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import argparse

# ==========================================
# ⚙️ 全局設定與樣式
# ==========================================
st.set_page_config(layout="wide", page_title="FaceSwap Evaluation Exp", page_icon=":koala:")

# 自定義 CSS 優化排版
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📂 資料處理函式
# ==========================================
def smart_find_image(base_dir, filename_hint):
    if not filename_hint or not os.path.exists(base_dir): return None
    name_no_ext = os.path.splitext(filename_hint)[0]
    candidates = [
        f"0_{filename_hint}", 
        filename_hint, 
        f"0_{name_no_ext}.png", 
        f"{name_no_ext}.png",
        f"0_{name_no_ext}.jpg"
    ]
    for cand in candidates:
        path = os.path.join(base_dir, cand)
        if os.path.exists(path): return path
    return None

def find_target_by_prompt(base_dir, prompt):
    if not prompt or not os.path.exists(base_dir): return None
    def normalize_quotes(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    target_normalized = normalize_quotes(prompt)
    for filename in os.listdir(base_dir):
        file_no_ext = os.path.splitext(filename)[0]
        if normalize_quotes(file_no_ext) == target_normalized:
            return os.path.join(base_dir, filename)
    return None

def normalize_fga(raw_score):
    """將 FGA Score 正規化到 0~1 (假設原始分佈約 1~5)"""
    if raw_score is None: return 0.0
    val = float(raw_score)
    return min(max((val - 1.0) / 4.0, 0.0), 1.0)

def calculate_weighted_score(item, weights, mode='swap'):
    """
    動態計算總分
    mode: 'swap' (換臉後) 或 't2i' (原圖)
    """
    # 根據 mode 決定欄位後綴
    suffix = "_t2i" if mode == 't2i' else ""
    
    # 1. Expression (Binary 0/1)
    k_exp = f"expression_correct{suffix}"
    s_exp = float(item.get(k_exp, 0) or 0)
    
    # 2. Scenario (Score 0.0~1.0)
    k_scen = f"scenario_score{suffix}"
    s_scen = float(item.get(k_scen, 0.0) or 0.0)

    # 3. Pose (Binary 0/1)
    k_pose = f"pose_correct{suffix}"
    s_pose = float(item.get(k_pose, 0) or 0)

    # 4. ID Similarity (Cosine 0.0~1.0)
    k_id = "id_similarity_t2i" if mode == 't2i' else "id_similarity"
    s_id = float(item.get(k_id, 0.0) or 0.0)
    s_id = max(0.0, min(s_id, 1.0))

    # 計算加權平均
    total_score = (
        (s_exp  * weights['expression']) +
        (s_scen * weights['scenario']) +
        (s_pose * weights['pose']) +
        (s_id   * weights['id'])
    )
    
    return round(total_score, 2), {
        'Exp': s_exp, 'Scen': s_scen, 'Pose': s_pose, 'ID': s_id
    }

@st.cache_data
def load_data(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def FGA_normalize(raw_score):
    """
    將 FGA-BLIP2 的 Overall Score 進行正規化。
    Args:
        raw_score (float): 模型輸出的原始 Logit 分數 (1~5)
        
    """
    # 1. 定義邊界 (根據官方圖表)
    MIN_VAL = 1.0
    MAX_VAL = 5.0
    
    # 3. Min-Max 正規化公式
    normalized_score = ((raw_score - MIN_VAL) / (MAX_VAL - MIN_VAL))
    
    return round(normalized_score, 2)
# ==========================================
# 🖥️ 主程式
# ==========================================
# ... (前面的 import 與函式保持不變) ...

def main(json_path, method, path_map):
    # 支援透過 Sidebar 覆蓋參數
    with st.sidebar:
        st.title("🎛️ Configuration")
        
        # A. 方法選擇
        try:
            default_idx = list(path_map.keys()).index(method)
        except ValueError:
            default_idx = 0
            
        selected_method = st.selectbox("Select Method", list(path_map.keys()), index=default_idx)
        SWAPPED_DIR, T2I_DIR = path_map.get(selected_method, path_map['pixart'])
        REF_DIR = "./faceswap_results/reference"
        
        # B. 權重調整
        st.subheader("⚖️ Weight Tuning")
        w_exp = st.slider("Expression Weight", 0.0, 1.0, 0.17, 0.01)
        w_scen = st.slider("Scenario Weight", 0.0, 1.0, 0.17, 0.01)
        w_pose = st.slider("Pose Weight", 0.0, 1.0, 0.34, 0.01)
        w_id = st.slider("ID Weight", 0.0, 1.0, 0.32, 0.01)
        
        weights = {'expression': w_exp, 'scenario': w_scen, 'pose': w_pose, 'id': w_id}

        # C. 選擇要比較的額外指標
        st.subheader("📊 Metrics to Compare")
        metric_options = ['CLIP T2I', 'CLIP Structure (I2I)', 'DINO Structure', 'FGA Score', 'CLIP I2I']
        selected_metrics = st.multiselect("Select Deep Metrics:", metric_options, default=['CLIP T2I', 'FGA Score', 'CLIP Structure (I2I)', 'CLIP I2I', 'DINO Structure'])

    # --- 2. 資料載入與計算 ---
    json_path_auto = os.path.join(SWAPPED_DIR, "evaluated_metrics.json")
    if os.path.exists(json_path_auto):
        json_path = json_path_auto

    raw_data = load_data(json_path)
    if not raw_data:
        st.error(f"❌ Failed to load JSON from {json_path}")
        return

    # 計算分數並加入 DataFrame
    processed_data = []
    for item in raw_data:
        score_swap, comps_swap = calculate_weighted_score(item, weights, 'swap')
        score_t2i, comps_t2i = calculate_weighted_score(item, weights, 't2i')
        
        item['score_swap'] = score_swap
        item['score_t2i'] = score_t2i
        
        for k, v in comps_swap.items():
            item[f'swap_{k}'] = v
        for k, v in comps_t2i.items():
            item[f't2i_{k}'] = v
            
        item['comps_swap'] = comps_swap
        item['comps_t2i'] = comps_t2i
        
        processed_data.append(item)
    
    df = pd.DataFrame(processed_data)

    # --- 3. Main Dashboard ---
    st.title(f"🧪 Evaluation Lab: {selected_method.capitalize()}")
    
    # 頂部概覽
    col1, col2, col3 = st.columns(3)
    avg_swap = df['score_swap'].mean()
    avg_t2i = df['score_t2i'].mean()
    delta = avg_swap - avg_t2i
    
    col1.metric("Avg Swapped Score", f"{avg_swap:.3f}", f"{delta:.3f}")
    col2.metric("Avg T2I (Baseline)", f"{avg_t2i:.3f}")
    col3.metric("Total Samples", len(df))

    st.divider()

    # ==========================================
    # 📊 分佈分析與異常值偵測 (Distribution Analysis)
    # ==========================================
    st.header("📉 Low Performance Analysis (Standard Deviation)")
    
    # 計算統計數據
    stats_swap_mean = df['score_swap'].mean()
    stats_swap_std = df['score_swap'].std()
    stats_t2i_mean = df['score_t2i'].mean()
    stats_t2i_std = df['score_t2i'].std()

    # 定義閥值
    th_swap_1std = stats_swap_mean - stats_swap_std
    th_swap_2std = max(stats_swap_mean - (2 * stats_swap_std), 0)
    th_t2i_1std = stats_t2i_mean - stats_t2i_std
    th_t2i_2std = max(stats_t2i_mean - (2 * stats_t2i_std), 0)

    # 顯示統計數據面板
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.info(f"**Swapped Image Stats**\n\nMean: {stats_swap_mean:.3f} | Std: {stats_swap_std:.3f}\n\n-1σ: {th_swap_1std:.3f}\n\n-2σ: {th_swap_2std:.3f}")
    with stat_col2:
        st.info(f"**T2I Source Stats**\n\nMean: {stats_t2i_mean:.3f} | Std: {stats_t2i_std:.3f}\n\n-1σ: {th_t2i_1std:.3f}\n\n-2σ: {th_t2i_2std:.3f}")

    # 篩選異常圖片
    st.subheader("⚠️ Outliers Detection")
    outlier_type = st.radio("Select Analysis Mode:", ["Swapped Result Outliers", "T2I Source Outliers"], horizontal=True)

    if outlier_type == "Swapped Result Outliers":
        df_extreme = df[df['score_swap'] <= th_swap_2std]
        df_bad = df[(df['score_swap'] > th_swap_2std) & (df['score_swap'] < th_swap_1std)]
        target_col = 'score_swap'
    else:
        df_extreme = df[df['score_t2i'] <= th_t2i_2std]
        df_bad = df[(df['score_t2i'] > th_t2i_2std) & (df['score_t2i'] < th_t2i_1std)]
        target_col = 'score_t2i'

    # 顯示分組結果
    tab_extreme, tab_bad, tab_all = st.tabs([
        f"🚨 Extreme Low (< -2σ) [{len(df_extreme)}]", 
        f"🔻 Below Average (< -1σ) [{len(df_bad)}]",
        "📋 Full List"
    ])

    with tab_extreme:
        if not df_extreme.empty:
            st.dataframe(df_extreme[['id', target_col, 'prompt']].style.format({target_col: "{:.3f}"}))
            df_filtered = df_extreme
        else:
            st.success("No samples found below 2 standard deviations! (Good consistency)")
            df_filtered = df 

    with tab_bad:
        if not df_bad.empty:
            st.dataframe(df_bad[['id', target_col, 'prompt']].style.format({target_col: "{:.3f}"}))
            if df_filtered.empty or df_filtered is df:
                 df_filtered = df_bad
        else:
            st.write("No samples in this range.")

    with tab_all:
        df_filtered_all = df.sort_values(target_col)
        st.dataframe(df_filtered_all[['id', target_col, 'prompt']])
        if 'df_filtered' not in locals() or (df_filtered is df):
             df_filtered = df_filtered_all

    st.divider()

    # ==========================================
    # 🕵️ Manual Failure Analysis
    # ==========================================
    st.subheader("🕵️ Manual Filter & Inspection")
    
    filter_metrics = ['score_swap', 'score_t2i', 'swap_Exp', 'swap_Scen', 'swap_Pose', 'swap_ID']
    
    f_c1, f_c2 = st.columns([1, 2])
    with f_c1:
        focus_metric = st.selectbox("Select Metric to Inspect:", filter_metrics)
    
    with f_c2:
        c_mean = df[focus_metric].mean()
        c_std = df[focus_metric].std()
        f_threshold = st.slider(f"Filter samples where {focus_metric} < X:", 0.0, 1.0, float(max(0.0, c_mean - c_std)))
        st.caption(f"Metric Mean: {c_mean:.2f} | Std: {c_std:.2f}")

    use_auto_filter = st.checkbox("Use data from 'Outliers Detection' tabs above?", value=True)
    
    if use_auto_filter and not df_filtered.empty:
        df_sorted = df_filtered.sort_values(focus_metric)
        st.info(f"Displaying {len(df_sorted)} samples from the Outlier Tabs above.")
    else:
        df_manual = df[df[focus_metric] <= f_threshold].sort_values(focus_metric)
        df_sorted = df_manual
        st.info(f"Displaying {len(df_sorted)} samples based on manual slider threshold.")

    st.divider()

    # --- 4. 詳細案例分析 (Case Inspection) ---
    st.header("🔍 Individual Case Inspection")
    
    if df_sorted.empty:
        st.warning("No samples match the current filter criteria.")
    else:
        selected_case_str = st.selectbox(
            "Select Case:", 
            df_sorted.apply(lambda x: f"{x['id']} | Swap: {x['score_swap']:.2f} | T2I: {x['score_t2i']:.2f} | Prompt: {x['prompt'][:40]}...", axis=1)
        )
        
        case_id = int(selected_case_str.split(" | ")[0])
        item = df[df['id'] == case_id].iloc[0]

        # --- A. 圖片展示區 ---
        col_imgs = st.columns(3) 
        
        ref_path = smart_find_image(REF_DIR, f"{str(item['id']).split('_')[0] if '_' in str(item['id']) else '0'}.png")
        if not ref_path: ref_path = os.path.join(REF_DIR, "0.png")
        
        t2i_path = find_target_by_prompt(T2I_DIR, item.get('prompt'))
        swap_path = smart_find_image(SWAPPED_DIR, item.get('image', '').strip())

        with col_imgs[0]:
            st.caption("1. Reference (ID)")
            if ref_path and os.path.exists(ref_path): st.image(Image.open(ref_path), use_container_width=True)
            else: st.warning("Ref not found")

        with col_imgs[1]:
            st.caption(f"2. T2I Source (Score: {item['score_t2i']})")
            if t2i_path: st.image(Image.open(t2i_path), use_container_width=True)
            else: st.warning("T2I not found")

        with col_imgs[2]:
            st.caption(f"3. Swapped Result (Score: {item['score_swap']})")
            if swap_path: st.image(Image.open(swap_path), use_container_width=True)
            else: st.warning("Swap not found")

        # --- B. Metric Breakdown (Grouped Bar Chart) ---
        st.markdown("#### 📊 Metric Breakdown")
        
        radar_data = []
        categories = ['Expression', 'Scenario', 'Pose', 'ID Similarity']
        
        vals_t2i = [item['t2i_Exp'], item['t2i_Scen'], item['t2i_Pose'], item['t2i_ID']]
        for c, v in zip(categories, vals_t2i):
            radar_data.append({'Category': c, 'Value': v, 'Type': 'T2I (Baseline)'})
            
        vals_swap = [item['swap_Exp'], item['swap_Scen'], item['swap_Pose'], item['swap_ID']]
        for c, v in zip(categories, vals_swap):
            radar_data.append({'Category': c, 'Value': v, 'Type': 'Swapped'})

        df_radar = pd.DataFrame(radar_data)
        
        chart = alt.Chart(df_radar).mark_bar().encode(
            x=alt.X('Category:N', title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Value:Q', title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Type:N', legend=alt.Legend(orient="top", title=None)),
            xOffset='Type:N',
            tooltip=['Category', 'Type', alt.Tooltip('Value', format='.2f')]
        ).properties(
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

        # --- C. 詳細指標表格 & 推理 ---
        st.subheader("📝 Deep Dive Analysis")
        tab1, tab2, tab3 = st.tabs(["🔢 Metrics Table", "🤖 VLM Reasoning", "🎨 FGA Token Heatmap"])

        with tab1:
            def get_val(key, default=0.0): return float(item.get(key, default) or 0.0)
            
            metrics_comp = {
                "Metric": [], "T2I (Orig)": [], "Swapped": [], "Diff": []
            }

            # [NEW] 1. 加入你要求的總分 (Weighted Total Score)
            metrics_comp["Metric"].append("**🏆 Weighted Total Score**")
            metrics_comp["T2I (Orig)"].append(f"{item['score_t2i']:.2f}")
            metrics_comp["Swapped"].append(f"{item['score_swap']:.2f}")
            metrics_comp["Diff"].append(f"{item['score_swap'] - item['score_t2i']:+.2f}")
            
            # 2. 顯示主要的 4 項指標比較
            core_map = {
                'Expression': ('t2i_Exp', 'swap_Exp'),
                'Scenario': ('t2i_Scen', 'swap_Scen'),
                'Pose': ('t2i_Pose', 'swap_Pose'),
                'ID Similarity': ('t2i_ID', 'swap_ID')
            }
            
            for m_name, (k_t2i, k_swap) in core_map.items():
                val_t2i = item[k_t2i]
                val_swap = item[k_swap]
                metrics_comp["Metric"].append(f"**{m_name}**") 
                metrics_comp["T2I (Orig)"].append(f"{val_t2i:.2f}")
                metrics_comp["Swapped"].append(f"{val_swap:.2f}")
                metrics_comp["Diff"].append(f"{val_swap - val_t2i:+.2f}")

            # 3. 顯示額外的 Deep Metrics
            real_key_map = {
                 'CLIP T2I': ('clip_t2i_orig', 'clip_t2i_swap'),
                 'CLIP Structure (I2I)': (None, 'clip_struct_score'),
                 'DINO Structure': (None, 'dino_struct_score'),
                 'FGA Score': ('fga_orig_score', 'fga_swap_score'),
                 'CLIP I2I': ('clip_id_t2i', 'clip_id_swap')
            }

            for m_name in selected_metrics:
                key_t2i, key_swap = real_key_map.get(m_name, (None, None))
                val_t2i = get_val(key_t2i) if key_t2i else np.nan
                val_swap = get_val(key_swap) if key_swap else np.nan
                
                # [NEW] 這裡加入 FGA Normalization 邏輯
                display_name = m_name
                if m_name == 'FGA Score':
                    val_t2i = FGA_normalize(val_t2i) if not np.isnan(val_t2i) else np.nan
                    val_swap = FGA_normalize(val_swap) if not np.isnan(val_swap) else np.nan
                    display_name = "FGA Score (Norm)" # 標註已正規化

                metrics_comp["Metric"].append(display_name)
                metrics_comp["T2I (Orig)"].append(f"{val_t2i:.3f}" if not np.isnan(val_t2i) else "-")
                metrics_comp["Swapped"].append(f"{val_swap:.3f}" if not np.isnan(val_swap) else "-")
                
                if not np.isnan(val_t2i) and not np.isnan(val_swap):
                    metrics_comp["Diff"].append(f"{val_swap - val_t2i:+.3f}")
                else:
                    metrics_comp["Diff"].append("-")

            st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### T2I Reasoning")
                st.info(f"**Expression:** {item.get('vlm_expression_t2i', 'N/A')}\n\n**Scenario Reasoning:**\n{item.get('scenario_reasoning_t2i', 'N/A')}")
            with c2:
                st.markdown("#### Swapped Reasoning")
                st.info(f"**Expression:** {item.get('vlm_expression', 'N/A')}\n\n**Scenario Reasoning:**\n{item.get('scenario_reasoning', 'N/A')}")

        with tab3:
            st.markdown("### 🧬 Fine-grained Token Alignment Comparison")
            st.caption("比較 T2I 原圖與換臉後 (Swapped) 每個 Token 的關注度分數變化。")

            # 1. 讀取兩組 Token 資料
            tokens_t2i = item.get('fga_orig_tokens', [])
            tokens_swap = item.get('fga_swap_tokens', [])
            
            # Fallback 機制 (如果 JSON 結構不同)
            if not tokens_swap: tokens_swap = item.get('all_token_scores', [])

            if tokens_t2i or tokens_swap:
                dfs = []
                
                # 處理 T2I 資料
                if tokens_t2i:
                    df_t = pd.DataFrame(tokens_t2i, columns=['Token', 'Score'])
                    df_t['Type'] = '1. T2I Source' # 加數字是為了強制排序讓 T2I 在左邊
                    df_t['Idx'] = range(len(df_t))
                    dfs.append(df_t)
                
                # 處理 Swapped 資料
                if tokens_swap:
                    df_s = pd.DataFrame(tokens_swap, columns=['Token', 'Score'])
                    df_s['Type'] = '2. Swapped Result'
                    df_s['Idx'] = range(len(df_s))
                    dfs.append(df_s)
                
                # 合併資料
                df_all = pd.concat(dfs)
                
                # 建立 Y 軸標籤 (確保順序正確)
                df_all['Label'] = df_all.apply(lambda x: f"{x['Idx']}: {x['Token']}", axis=1)

                # 計算圖表高度 (根據最長的那個序列)
                max_len = len(df_all) / len(dfs) if dfs else 0
                chart_height = max(400, int(max_len * 25))

                # 繪製對比圖
                charts = alt.Chart(df_all).mark_bar().encode(
                    # X 軸: 分數
                    x=alt.X('Score:Q', title="FGA Attention Score", scale=alt.Scale(domain=[0, max(df_all['Score'].max(), 1.0)])),
                    
                    # Y 軸: Token 序列 (使用 Idx 排序)
                    y=alt.Y('Label:N', title=None, sort=alt.EncodingSortField(field='Idx', order='ascending')),
                    
                    # 顏色: 低分紅，高分綠
                    color=alt.condition(alt.datum.Score < 0.1, alt.value('#ff4b4b'), alt.value('#00cc96')),
                    
                    # Tooltip
                    tooltip=['Type', 'Token', 'Score']
                ).properties(
                    width=350, # 每個子圖的寬度
                    height=chart_height
                ).facet(
                    # 分欄: 依照 Type 分成左右兩欄
                    column=alt.Column('Type:N', title=None, header=alt.Header(labelFontSize=14, labelFontWeight='bold'))
                ).resolve_scale(
                    x='shared', # 共用 X 軸刻度方便比較
                    y='shared'  # 共用 Y 軸確保對齊
                )

                st.altair_chart(charts, use_container_width=False) # False 避免擠壓，讓它使用設定的 width
            
            else:
                st.warning("⚠️ No token-level scores found in the JSON data for this case.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='./gt.json')
    parser.add_argument('--method', type=str, default='pixart')
    args = parser.parse_args()
    
    path_map = {
        'pixart': ('./faceswap_results/pixart', './pixart_outputs'),
        'janus': ('./faceswap_results/janus', './pixart_outputs'), 
        'infinity': ('./faceswap_results/infinity', './pixart_outputs'),
        'showo2': ('./faceswap_results/showo2', './pixart_outputs')
    }
    
    main(args.json, args.method, path_map)