import argparse
import json
import os
import pandas as pd
import numpy as np

# ==========================================
# 權重設定 (Total = 1.0)
# ==========================================
# 這些權重對應到 JSON 中的欄位名稱
WEIGHTS = {
    'expression': 0.17,
    'scenario':   0.17,
    # 'gender':     0.17, # Gender 暫時不計分
    'pose':       0.34,
    'id':         0.32
}

def calculate_single_item_score(item, prefix):
    """
    計算單個項目的加權總分。
    prefix: 'swap_' 或 't2i_' (對應 JSON 欄位前綴)
    """
    # 1. Expression (0 or 1)
    # key: expression_correct (swap) / expression_correct_t2i (t2i)
    # 注意：原本 JSON 欄位命名不完全統一，這裡做個對應
    if prefix == 't2i_':
        s_exp = float(item.get('expression_correct_t2i', 0))
    else:
        s_exp = float(item.get('expression_correct', 0)) # Swap 通常用這個欄位名
    
    # 2. Scenario (0.0 ~ 1.0)
    # key: swap_scenario_score / t2i_scenario_score
    s_scen = float(item.get(f'{prefix}scenario_score', 0.0))
    
    # 3. Pose (0 or 1)
    # key: swap_pose_match / t2i_pose_match (根據之前的 Pose Script 修改)
    # 或是 swap_pose_correct / t2i_pose_correct
    # 這裡嘗試讀取兩種可能的命名
    pose_key = f'{prefix}pose_match'
    if pose_key not in item:
        pose_key = f'{prefix}pose_correct'
    s_pose = float(item.get(pose_key, 0))
    
    # 4. ID Similarity (0.0 ~ 1.0)
    # key: swap_id_similarity / t2i_id_similarity
    id_key = f'{prefix}id_similarity'
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
    final_score = (
        (s_exp  * WEIGHTS['expression']) +
        (s_scen * WEIGHTS['scenario']) +
        (s_pose * WEIGHTS['pose']) +
        (s_id   * WEIGHTS['id'])
    )
    
    return round(final_score, 4), s_exp, s_scen, s_pose, s_id

def main(json_path, task_name, mode='full'):
    if not os.path.exists(json_path):
        print(f"❌ Error: 找不到檔案 {json_path}")
        return

    print(f"📂 Reading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 用來收集數據做 DataFrame 分析
    df_data = []

    print(f"📊 Calculating scores for task [{task_name}] ({len(data_list)} images) | Mode: {mode.upper()}")
    
    for item in data_list:
        row_data = {
            'Prompt': item.get('prompt', '')[:30] + '...',
        }

        # --- 計算 T2I 分數 (所有模式) ---
        score_t2i, exp_t2i, scen_t2i, pose_t2i, id_t2i = calculate_single_item_score(item, prefix='t2i_')
        item['t2i_final_score'] = score_t2i
        
        row_data.update({
            'Score_T2I': score_t2i,
            'Exp_T2I': exp_t2i,
            'Scen_T2I': scen_t2i,
            'Pose_T2I': pose_t2i,
            'ID_T2I': id_t2i
        })

        # --- 計算 Swap 分數 (僅 Full 模式) ---
        if mode == 'full':
            score_swap, exp_swap, scen_swap, pose_swap, id_swap = calculate_single_item_score(item, prefix='swap_') # swap 通常欄位前綴較亂，這裡假設我們已統一
            # 如果欄位是舊版命名 (無 swap_ 前綴)，需手動調整 `calculate_single_item_score` 或確保 JSON 欄位一致
            # 這裡我們假設 pose_eval 和 id_eval 腳本已經產生了標準的 `swap_` 前綴欄位，
            # 若 id 是 `swap_id_similarity`，pose 是 `swap_pose_match`
            
            item['swap_final_score'] = score_swap
            
            row_data.update({
                'Score_Swap': score_swap,
                'Exp_Swap': exp_swap,
                'Scen_Swap': scen_swap,
                'Pose_Swap': pose_swap,
                'ID_Swap': id_swap
            })
            
        df_data.append(row_data)

    # === 1. 更新原本的 JSON ===
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"✅ Scores updated in: {json_path}")

    # === 2. 產生簡易終端機報表 ===
    if not df_data:
        print("⚠️ No data to analyze.")
        return

    df = pd.DataFrame(df_data)
    
    print("\n" + "="*80)
    print(f"🏆 Score Report: {task_name}")
    print("="*80)
    
    if mode == 'full':
        print(f"{'Metric':<20} | {'Swapped Result':<15} | {'T2I Baseline':<15}")
        print("-" * 60)
        print(f"{'Final Score (Avg)':<20} | {df['Score_Swap'].mean():<15.4f} | {df['Score_T2I'].mean():<15.4f}")
        print(f"{'ID Similarity':<20} | {df['ID_Swap'].mean():<15.4f} | {df['ID_T2I'].mean():<15.4f}")
        print(f"{'Expression Acc':<20} | {df['Exp_Swap'].mean()*100:<15.1f}% | {df['Exp_T2I'].mean()*100:<15.1f}%")
        print(f"{'Scenario Acc':<20} | {df['Scen_Swap'].mean()*100:<15.1f}% | {df['Scen_T2I'].mean()*100:<15.1f}%")
        print(f"{'Pose Acc':<20} | {df['Pose_Swap'].mean()*100:<15.1f}% | {df['Pose_T2I'].mean()*100:<15.1f}%")
        
        # 顯示 Top 3
        print(f"\n🌟 Top 3 Best Swapped Images:")
        print(df.sort_values(by='Score_Swap', ascending=False).head(3)[['Prompt', 'Score_Swap']].to_string(index=False))

    else:
        # T2I Only 報表
        print(f"{'Metric':<20} | {'T2I Baseline':<15}")
        print("-" * 40)
        print(f"{'Final Score (Avg)':<20} | {df['Score_T2I'].mean():<15.4f}")
        print(f"{'ID Similarity':<20} | {df['ID_T2I'].mean():<15.4f}")
        print(f"{'Expression Acc':<20} | {df['Exp_T2I'].mean()*100:<15.1f}%")
        print(f"{'Scenario Acc':<20} | {df['Scen_T2I'].mean()*100:<15.1f}%")
        print(f"{'Pose Acc':<20} | {df['Pose_T2I'].mean()*100:<15.1f}%")

        # 顯示 Top 3
        print(f"\n🌟 Top 3 Best T2I Images:")
        print(df.sort_values(by='Score_T2I', ascending=False).head(3)[['Prompt', 'Score_T2I']].to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Final Weighted Scores with Mode Selection")
    parser.add_argument("--json", type=str, default="metadata.json", help="Input JSON file path")
    parser.add_argument("--name", type=str, default="pixart", help="Name of the method/task")
    # 為了保持指令一致性，這裡接受但忽略路徑參數
    parser.add_argument("--swap", type=str, default="", help="(Ignored) Swapped dir")
    parser.add_argument("--t2i", type=str, default="", help="(Ignored) T2I dir")
    parser.add_argument("--ref", type=str, default="", help="(Ignored) Ref dir")
    # 新增 mode 參數
    parser.add_argument("--mode", type=str, default="full", choices=["full", "t2i"], help="Evaluation Mode")
    
    args = parser.parse_args()

    main(args.json, args.name, mode=args.mode)