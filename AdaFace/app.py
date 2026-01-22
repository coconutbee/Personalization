import streamlit as st
import pandas as pd
import os

# --- 頁面設定 ---
st.set_page_config(page_title="Face Swap Evaluation", layout="wide")

# CSS 優化：讓圖片標題置中，增加卡片效果
st.markdown("""
    <style>
    div[data-testid="stImage"] {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
        padding: 5px;
        background-color: #f9f9f9;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧩 Face Personalization Evaluation")
st.markdown("### ID Retention & Pose Transfer Check")
st.info("邏輯：檢查 **Output** 是否保留了 **Source** 的 ID，並套用了 **Target** 的 Pose/輪廓。")

# --- 側邊欄：設定 ---
with st.sidebar:
    st.header("📂 資料夾路徑設定")
    
    # 預設路徑
    base_path = '/media/ee303/disk1/Personalization'
    default_csv = 'id_similarity_results.csv'
    
    csv_path = st.text_input("CSV Path", value=default_csv)
    source_dir = st.text_input("Source Dir (ID Provider)", value=f"{base_path}/benchmark_data/source")
    target_dir = st.text_input("Target Dir (Pose Provider)", value=f"{base_path}/benchmark_data/target")
    output_dir = st.text_input("Output Dir (Result)", value=f"{base_path}/output")

    st.divider()
    st.header("🔍 篩選條件")
    
    # 排序
    sort_order = st.radio("排序方式", 
                          ["ID 相似度：低 -> 高 (找 ID 遺失)", 
                           "ID 相似度：高 -> 低 (找成功案例)"])
    
    # 分數過濾
    min_score, max_score = st.slider("ID Similarity 過濾", -1.0, 1.0, (-1.0, 1.0))
    
    st.caption("註：CSV 分數代表 Source 與 Output 的 ID 相似度")

# --- 讀取資料 ---
if not os.path.exists(csv_path):
    st.error(f"找不到 CSV: {csv_path}，請確認計算腳本已執行。")
    st.stop()

df = pd.read_csv(csv_path)

# 過濾與排序
df_filtered = df[(df['id_similarity'] >= min_score) & (df['id_similarity'] <= max_score)].copy()

ascending = True if "低 -> 高" in sort_order else False
df_filtered = df_filtered.sort_values(by='id_similarity', ascending=ascending)

# --- 頂部統計 ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("總圖片數", len(df))
c2.metric("顯示圖片數", len(df_filtered))
c3.metric("平均 ID 相似度", f"{df_filtered['id_similarity'].mean():.4f}")
pass_rate = (df_filtered['id_similarity'] >= 0.3).mean() * 100
c4.metric("ID 合格率 (>=0.3)", f"{pass_rate:.1f}%")

st.divider()

# --- 主畫面列表 ---
for index, row in df_filtered.iterrows():
    fname = row['filename']
    score = row['id_similarity']
    
    # 分數顯示顏色與文字
    if pd.isna(score):
        score_text = "⚠️ Face Not Found"
        bar_color = "gray"
        score_val = 0.0
    else:
        score_text = f"ID Sim: {score:.4f}"
        score_val = max(0.0, min(1.0, score))
        if score > 0.5: bar_color = "green"
        elif score > 0.3: bar_color = "orange"
        else: bar_color = "red"

    # 使用 Container 包覆每一列
    with st.container():
        st.markdown(f"**Filename:** `{fname}`")
        
        # 建立三欄：Source | Target | Output
        col_src, col_tgt, col_out = st.columns([1, 1, 1.2]) # Output 欄位稍微大一點
        
        # 1. ID Source
        with col_src:
            src_path = os.path.join(source_dir, fname)
            if os.path.exists(src_path):
                st.image(src_path, caption="Source (ID)", use_container_width=True)
            else:
                st.warning("No Source")

        # 2. Pose Target
        with col_tgt:
            tgt_path = os.path.join(target_dir, fname)
            if os.path.exists(tgt_path):
                st.image(tgt_path, caption="Target (Pose/Contour)", use_container_width=True)
            else:
                st.warning("No Target")

        # 3. Output Result
        with col_out:
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path):
                st.image(out_path, caption="Output (Result)", use_container_width=True)
                
                # 在 Output 下方顯示分數條
                if not pd.isna(score):
                    st.markdown(f"<p style='color:{bar_color}; font-weight:bold; margin:0;'>{score_text}</p>", unsafe_allow_html=True)
                    st.progress(score_val)
                else:
                    st.error("Detection Failed")
            else:
                st.warning("No Output")

    st.markdown("---")