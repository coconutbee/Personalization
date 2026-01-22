import argparse
import pandas as pd
import os
import re

# 1. 定義映射規則
MAPPING_RULES = {
    'happy': ['happy', 'smile', 'smiles', 'grinning', 'smiling', 'laugh', 'giggles', 'joy', 'smirk', 'joyful', 'enjoy'],
    'surprise': ['surprised', 'surprise', 'amazed', 'astonished', 'skeptical', 'shock'],
    'confuse': ['confuse', 'forward', 'puzzled', 'questioning', 'thoughtful', 'confused'],
    'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene', 'calm'],
    'sad': ['sad', 'sadness', 'crying', 'gloomy', 'depressed'], 
    'others': ['others']
}

def clean_think_content(text):
    """
    專門用來移除 <think>...</think> 區塊的函數
    """
    if pd.isna(text):
        return ""
    text_str = str(text)
    # 使用 re.DOTALL 確保 . 可以匹配換行符號
    cleaned = re.sub(r'<think>.*?</think>', '', text_str, flags=re.DOTALL)
    return cleaned.strip()

def get_standard_label(val):
    """
    通用映射函數
    """
    if pd.isna(val):
        return "unknown"
    
    # 這裡傳入的 val 已經是被 clean_think_content 清洗過的，所以很乾淨
    val_str = str(val).lower()
    val_clean = re.sub(r'[^\w\s]', ' ', val_str) 

    for label, keywords in MAPPING_RULES.items():
        for kw in keywords:
            kw_lower = kw.lower()
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', val_clean):
                return label
            
    return "others"

def rigorous_clean(text):
    """ 強力清洗路徑字串用於 merge key """
    if pd.isna(text): return ""
    text = str(text)
    text = os.path.splitext(text)[0]
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text.lower()

def map_expression_results(csv_path, output_path):
    print(f"正在處理模型預測結果: {csv_path} ...")
    df = pd.read_csv(csv_path)

    # ==========================================
    # 核心修改：在映射之前，先清洗掉 <think> 標籤
    # 這樣寫入 CSV 的 'expression_internvl' 就會是乾淨的
    # ==========================================
    if 'expression_internvl' in df.columns:
        df['expression_internvl'] = df['expression_internvl'].apply(clean_think_content)

    # 對「已經清洗過」的結果應用映射
    df['mapped_expression'] = df['expression_internvl'].apply(get_standard_label)

    # 儲存結果 (此時 expression_internvl 欄位已經沒有 <think> 了)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    unmapped_count = len(df[df['mapped_expression'] == 'others'])
    print(f"-> 模型結果映射完成。歸類為 others 數量: {unmapped_count} / {len(df)}")

def final_attempt_merging(results_csv, gt_csv, output_csv):
    print(f"正在合併與比對 GT: {gt_csv} ...")
    df_results = pd.read_csv(results_csv)
    df_gt = pd.read_csv(gt_csv)

    # GT 映射
    df_gt['mapped_gt'] = df_gt['exps_ground_truth'].apply(get_standard_label)

    # 建立對照字典
    gt_map = {
        rigorous_clean(row['prompt']): row['mapped_gt'] 
        for _, row in df_gt.iterrows()
    }

    def get_ground_truth_label(prompt_val):
        clean_key = rigorous_clean(prompt_val)
        return gt_map.get(clean_key, "GT_NOT_FOUND")

    df_results['ground_truth_reference'] = df_results['prompt'].apply(get_ground_truth_label)

    # 計算正確率
    df_results['expression_correct'] = df_results.apply(
        lambda row: 1 if str(row['mapped_expression']).strip() == str(row['ground_truth_reference']).strip() else 0, 
        axis=1
    )

    valid_rows = df_results[df_results['ground_truth_reference'] != "GT_NOT_FOUND"]
    
    if len(valid_rows) > 0:
        accuracy = valid_rows['expression_correct'].sum() / len(valid_rows)
        print(f"-> 成功匹配 GT 數量: {len(valid_rows)} / {len(df_results)}")
        print(f"-> 最終精確度 (Accuracy): {accuracy:.2%}")
    else:
        print("-> 警告：沒有任何圖片成功匹配到 Ground Truth。")

    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"-> 結果已儲存至: {output_csv}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart')
    # method = 'pixart' # pixart, janus, infinity, showo2
    args = parser.parse_args()

    # 1. 映射並清洗模型結果
    map_expression_results(f'internVL_exps/internvl_exps_{args.method}.csv', 'mapped_results.csv')
    
    os.makedirs('internVL_exps', exist_ok=True)
    
    # 2. 合併 GT
    final_attempt_merging('internVL_exps/mapped_results.csv', 'gt.csv', f'internVL_exps/internVL_exps/{args.method}_exps.csv')