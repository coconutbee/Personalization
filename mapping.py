import pandas as pd
import os

def map_expression_results(csv_path, output_path):
    # 1. 定義映射規則
    mapping_rules = {
        'happy': ['happy', 'smile', 'smiles', 'grinning', 'smiling', 'laugh', 'giggles', 'joy', 'smirk', 'joyful'],
        'surprise': ['surprised', 'surprise', 'amazed', 'astonished', 'skeptical'],
        'confuse': ['confuse', 'forward', 'puzzled', 'questioning', 'thoughtful'],
        'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene'],
        'sad': ['sad', 'sadness', 'crying'], # 建議多加 sadness
        'others': ['others']
    }

    # 2. 讀取 CSV 資料
    df = pd.read_csv(csv_path)

    # 3. 執行映射轉換 (核心修改部分)
    def get_standard_label(val):
        if pd.isna(val):
            return "unknown"
        
        # A. 基礎清洗：轉小寫
        val_str = str(val).lower()
        
        # B. 進階清洗：移除標點符號 (只保留字母和空格)，這能解決 "smile." 或 "happy!" 的問題
        # 這樣做可以避免標點符號干擾比對
        val_clean = re.sub(r'[^\w\s]', ' ', val_str) 

        # C. 模糊比對邏輯
        # 遍歷每一個情緒類別 (Label)
        for label, keywords in mapping_rules.items():
            for kw in keywords:
                kw_lower = kw.lower()
                
                # 方法一：單字邊界比對 (比較嚴謹，避免 "enjoy" 被誤判為 "joy")
                # 使用 Regex 檢查是否包含完整的單字
                if re.search(r'\b' + re.escape(kw_lower) + r'\b', val_clean):
                    return label
                
                # 方法二：簡單字串包含 (如果你不在意 "enjoy" 變成 "joy"，用這個比較快且寬鬆)
                # if kw_lower in val_clean:
                #     return label

        return "unmapped"

    # 新增映射後的結果欄位
    df['mapped_expression'] = df['expression_internvl'].apply(get_standard_label)

    # 4. 儲存結果
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 檢查一下有多少比例是 unmapped
    unmapped_count = len(df[df['mapped_expression'] == 'unmapped'])
    print(f"處理完成！映射結果已儲存至: {output_path}")
    print(f"未映射(unmapped)數量: {unmapped_count} / {len(df)}")
import re

def rigorous_clean(text):
    """ 強力清洗字串：移除副檔名、標點符號、引號，並轉小寫 """
    if pd.isna(text): return ""
    text = str(text)
    # 1. 移除副檔名 (例如 .png)
    text = os.path.splitext(text)[0]
    # 2. 移除所有非字母數字的字元（包含各種引號、逗號、句號）
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    # 3. 轉小寫
    return text.lower()

def final_attempt_merging(results_csv, gt_csv, output_csv):
    df_results = pd.read_csv(results_csv)
    df_gt = pd.read_csv(gt_csv)

    # 建立一個「完全純淨版文字」到「Ground Truth」的字典
    # 使用標籤 'ground truth'
    gt_map = {
        rigorous_clean(row['prompt']): row['ground truth'] 
        for _, row in df_gt.iterrows()
    }

    # 執行比對
    def get_ground_truth(path_val):
        clean_key = rigorous_clean(path_val)
        # 嘗試匹配，若失敗則回傳 GT_NOT_FOUND
        return gt_map.get(clean_key, "GT_NOT_FOUND")

    df_results['ground_truth_reference'] = df_results['path'].apply(get_ground_truth)

    # 計算正確率：比較映射後的結果與參考標籤
    df_results['is_correct'] = df_results.apply(
        lambda row: 1 if str(row['mapped_expression']).strip().lower() == 
                         str(row['ground_truth_reference']).strip().lower() else 0, 
        axis=1
    )

    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    success_count = (df_results['ground_truth_reference'] != "GT_NOT_FOUND").sum()
    print(f"處理完成！成功匹配到 GT 的數量: {success_count} / {len(df_results)}")
    
    if success_count > 0:
        accuracy = df_results['is_correct'].sum() / success_count
        print(f"匹配成功部分的精確度 (Accuracy): {accuracy:.2%}")
# 使用範例
if __name__ == "__main__":
    map_expression_results('internvl_open_exps_showo2.csv', 'mapped_results.csv')
    os.makedirs('internVL_exps', exist_ok=True)
    final_attempt_merging('mapped_results.csv', 'gt.csv', 'internVL_exps/showo2_exps.csv')
    # pixart, janus, infinity, showo2