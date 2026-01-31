import argparse
import json
import os
import pandas as pd
import numpy as np

# ==========================================
# 權重設定
# ==========================================
WEIGHTS = {
    'expression': 0.17,
    'scenario':   0.17,
    # 'gender':     0.17,
    'pose':       0.34,
    'id':         0.32
}

def calculate_final_score(item):
    """
    計算單張圖片的加權總分 (滿分 100)
    """
    # 1. Expression (0 or 1)
    s_exp = float(item.get('expression_correct', 0))
    
    # 2. Scenario (0.0 ~ 1.0)
    s_scen = float(item.get('scenario_score', 0.0))
    
    # 3. Gender (0 or 1)
    # s_gen = float(item.get('gender_correct', 0))
    
    # 4. Pose (0 or 1)
    s_pose = float(item.get('pose_correct', 0))
    
    # 5. ID Similarity (字串轉浮點數，處理 None)
    raw_id = item.get('id_similarity')
    if raw_id is None:
        s_id = 0.0
    else:
        try:
            s_id = float(raw_id)
            # 確保 ID 分數在 0~1 之間
            s_id = max(0.0, min(s_id, 1.0))
        except ValueError:
            s_id = 0.0

    # === 加權計算 ===
    final_score = (
        (s_exp  * WEIGHTS['expression']) +
        (s_scen * WEIGHTS['scenario']) +
        # (s_gen  * WEIGHTS['gender']) +
        (s_pose * WEIGHTS['pose']) +
        (s_id   * WEIGHTS['id'])
    )
    
    # 轉換為百分制 (0~100)
    return round(final_score, 2), s_id

def main(json_path, output_csv):
    if not os.path.exists(json_path):
        print(f"❌ Error: 找不到檔案 {json_path}")
        return

    print(f"📂 Reading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 用來收集數據做 DataFrame 分析
    df_data = []

    print(f"📊 Calculating scores for {len(data_list)} images...")
    
    for item in data_list:
        score, clean_id_val = calculate_final_score(item)
        
        # 將分數寫回 JSON 物件 (方便後續查看)
        item['final_score'] = score
        
        # 收集數據到列表
        df_data.append({
            'Image': item.get('image', 'Unknown'),
            'Final_Score': score,
            'Exp (17%)': item.get('expression_correct', 0),
            'Scen (17%)': item.get('scenario_score', 0),
            # 'Gen (17%)': item.get('gender_correct', 0),
            'Pose (17%)': item.get('pose_correct', 0),
            'ID_Sim (32%)': clean_id_val,
            # 其他資訊方便除錯
            'VLM_Exp': item.get('vlm_expression', ''),
            'Mivolo_Gender': item.get('mivolo_gender', '')
        })

    # === 1. 更新原本的 JSON (加上 final_score) ===
    output_json_path = json_path.replace('.json', '_scored.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"✅ Updated JSON saved to: {output_json_path}")

    # === 2. 產生 CSV 報表 ===
    df = pd.DataFrame(df_data)
    
    # 計算平均指標
    avg_score = df['Final_Score'].mean()
    avg_id = df['ID_Sim (32%)'].mean()
    
    print("\n" + "="*50)
    print(f"🏆 Overall Benchmark Performance")
    print("="*50)
    print(f"Average Final Score : {avg_score:.2f} / 100")
    print(f"Average ID Similarity: {avg_id:.4f}")
    print(f"Expression Accuracy: {df['Exp (17%)'].mean()*100:.1f}%")
    print(f"Scenario Accuracy  : {df['Scen (17%)'].mean()*100:.1f}%")
    # print(f"Gender Accuracy    : {df['Gen (17%)'].mean()*100:.1f}%")
    print(f"Pose Accuracy      : {df['Pose (17%)'].mean()*100:.1f}%")
    print("-" * 50)
    
    # 顯示分數最高的前 5 名
    print("\n🌟 Top 5 Best Images:")
    print(df.sort_values(by='Final_Score', ascending=False).head(5)[['Image', 'Final_Score', 'ID_Sim (32%)']].to_string(index=False))

    # 顯示分數最低的前 5 名
    print("\n⚠️ Bottom 5 Worst Images:")
    print(df.sort_values(by='Final_Score', ascending=True).head(5)[['Image', 'Final_Score', 'ID_Sim (32%)']].to_string(index=False))

    # 存成 CSV
    df.to_csv(output_csv, index=False)
    print(f"\n📄 Detailed CSV report saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="gt.json", help="Input JSON file path")
    parser.add_argument("--csv", type=str, default="final_scores.csv", help="Output CSV file path")
    args = parser.parse_args()

    main(args.json, args.csv)