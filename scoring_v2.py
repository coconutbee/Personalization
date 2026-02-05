import argparse
import json
import os
import pandas as pd
import numpy as np

# ==========================================
# 權重設定 (Total = 1.0)
# ==========================================
WEIGHTS = {
    'expression': 0.17,
    'scenario':   0.17,
    # 'gender':     0.17, # Gender 暫時不計分
    'pose':       0.34,
    'id':         0.32
}

def calculate_final_score(item, is_t2i=False):
    """
    計算單張圖片的加權總分 (滿分 100)
    區分 Swapped Result 與 T2I Original Result
    """
    suffix = "_t2i" if is_t2i else ""
    
    # 1. Expression (0 or 1)
    # key: expression_correct / expression_correct_t2i
    s_exp = float(item.get(f'expression_correct{suffix}', 0))
    
    # 2. Scenario (0.0 ~ 1.0)
    # key: scenario_score / scenario_score_t2i
    s_scen = float(item.get(f'scenario_score{suffix}', 0.0))
    
    # 3. Gender (0 or 1) - 目前被註解掉，設為 0
    # s_gen = float(item.get(f'gender_correct{suffix}', 0))
    
    # 4. Pose (0 or 1)
    # key: pose_correct (假設 pose 只有一套標準，若 T2I 也有個別分數需調整)
    # 如果 T2I 沒算 pose，這邊預設為 0 或共用
    # 目前假設只有 swap 有 pose 分數，T2I 若無則為 0
    s_pose = float(item.get(f'pose_correct{suffix}', item.get('pose_correct', 0))) 
    
    # 5. ID Similarity (0.0 ~ 1.0)
    # key: id_similarity / t2i_id_similarity
    id_key = 't2i_id_similarity' if is_t2i else 'id_similarity'
    raw_id = item.get(id_key)
    
    if raw_id is None:
        s_id = 0.0
    else:
        try:
            s_id = float(raw_id)
            s_id = max(0.0, min(s_id, 1.0))
        except ValueError:
            s_id = 0.0

    # === 加權計算 (Score 0~1) ===
    # 注意：這裡權重總和應為 1.0，若移除 Gender 需確認其他權重是否補足
    # 目前 WEIGHTS 加總 = 0.17+0.17+0.34+0.32 = 1.0 (剛好)
    final_score = (
        (s_exp  * WEIGHTS['expression']) +
        (s_scen * WEIGHTS['scenario']) +
        # (s_gen  * WEIGHTS['gender']) +
        (s_pose * WEIGHTS['pose']) +
        (s_id   * WEIGHTS['id'])
    )
    
    # 取小數點後 4 位
    return round(final_score, 4), s_id

def main(json_path, task_name):
    if not os.path.exists(json_path):
        print(f"❌ Error: 找不到檔案 {json_path}")
        return

    print(f"📂 Reading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 用來收集數據做 DataFrame 分析
    df_data = []

    print(f"📊 Calculating scores for task [{task_name}] ({len(data_list)} images)...")
    
    for item in data_list:
        # --- 計算 Swapped Image 分數 ---
        score_swap, id_swap = calculate_final_score(item, is_t2i=False)
        item['swap_final_score'] = score_swap # 寫回 JSON
        
        # --- 計算 T2I Original Image 分數 (Baseline) ---
        score_t2i, id_t2i = calculate_final_score(item, is_t2i=True)
        item['t2i_final_score'] = score_t2i # 寫回 JSON

        # 收集數據到列表 (只收集 Swap 的詳細數據做分析，T2I 僅做對照)
        df_data.append({
            'Prompt': item.get('prompt', '')[:30] + '...', # 簡略顯示
            'Score_Swap': score_swap,
            'Score_T2I': score_t2i,
            'Exp': item.get('expression_correct', 0),
            'Scen': item.get('scenario_score', 0),
            'Pose': item.get('pose_correct', 0),
            'ID': id_swap,
        })

    # === 1. 更新原本的 JSON ===
    # 直接覆寫回原本的 json 檔案
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"✅ Scores updated in: {json_path}")

    # === 2. 產生簡易終端機報表 ===
    if not df_data:
        print("⚠️ No data to analyze.")
        return

    df = pd.DataFrame(df_data)
    
    # 計算平均指標
    avg_score_swap = df['Score_Swap'].mean()
    avg_score_t2i = df['Score_T2I'].mean()
    
    print("\n" + "="*60)
    print(f"🏆 Score Report: {task_name}")
    print("="*60)
    print(f"{'Metric':<20} | {'Swapped Result':<15} | {'T2I Baseline':<15}")
    print("-" * 60)
    print(f"{'Final Score (Avg)':<20} | {avg_score_swap:<15.2f} | {avg_score_t2i:<15.2f}")
    print(f"{'ID Similarity':<20} | {df['ID'].mean():<15.4f} | {'-':<15}")
    print(f"{'Expression Acc':<20} | {df['Exp'].mean()*100:<15.1f}% | {'-':<15}")
    print(f"{'Scenario Acc':<20} | {df['Scen'].mean()*100:<15.1f}% | {'-':<15}")
    print(f"{'Pose Acc':<20} | {df['Pose'].mean()*100:<15.1f}% | {'-':<15}")
    print("=" * 60)
    
    # 顯示分數最高的前 3 名
    print(f"\n🌟 Top 3 Best Images ({task_name}):")
    print(df.sort_values(by='Score_Swap', ascending=False).head(3)[['Prompt', 'Score_Swap', 'ID']].to_string(index=False))

    # 顯示分數最低的前 3 名
    print(f"\n⚠️ Bottom 3 Worst Images ({task_name}):")
    print(df.sort_values(by='Score_Swap', ascending=True).head(3)[['Prompt', 'Score_Swap', 'ID']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Final Weighted Scores")
    parser.add_argument("--json", type=str, default="metadata.json", help="Input JSON file path (Score will be updated here)")
    parser.add_argument("--name", type=str, default="pixart", help="Name of the method/task")
    # 為了保持指令一致性，這裡可以接受但忽略 --swap, --t2i 參數
    parser.add_argument("--swap", type=str, default="", help="(Ignored) Swapped dir")
    parser.add_argument("--t2i", type=str, default="", help="(Ignored) T2I dir")
    
    args = parser.parse_args()

    main(args.json, args.name)