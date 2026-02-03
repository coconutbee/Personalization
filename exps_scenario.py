import argparse
import os
import json
import re
import torch
from tqdm import tqdm
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# ==========================================
# 工具函式：智慧搜尋檔案
# ==========================================
def try_find_file(base_path, filename_seed, extensions):
    """
    嘗試多種變體來尋找檔案 (解決 ' 和 ’ 以及 0_ 前綴問題)
    """
    if not filename_seed or not os.path.exists(base_path): 
        return None, None

    # 移除副檔名以便重新組合
    name_no_ext = os.path.splitext(filename_seed)[0]

    # 可能的檔名變體
    variants = [
        name_no_ext,                        # 原始 (e.g., "1")
        f"0_{name_no_ext}",                 # 前綴 (e.g., "0_1")
        name_no_ext.replace("'", "’"),      # 符號替換
    ]

    for text in variants:
        for ext in extensions:
            filename = f"{text}{ext}"
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                return full_path, filename
    
    return None, None

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 找尋對應的 T2I 原圖"""
    if not prompt or not os.path.exists(base_dir): return None, None
    
    valid_extensions = ['.png', '.jpg', '.jpeg']
    
    # 簡單的正規化
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    
    target_norm = normalize(prompt)
    
    # 1. 直接嘗試 (最快)
    path, name = try_find_file(base_dir, prompt, valid_extensions)
    if path: return path, name

    # 2. 遍歷目錄 (較慢但穩健，解決檔名截斷問題)
    for filename in os.listdir(base_dir):
        name_no_ext = os.path.splitext(filename)[0]
        if normalize(name_no_ext) == target_norm:
            return os.path.join(base_dir, filename), filename
            
    return None, None

# ==========================================
# 工具函式：提取分數
# ==========================================
def extract_score(text: str) -> float:
    """從 VLM 回覆中提取分數"""
    match = re.search(r"(?:Match )?Score[:\s\n*]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return min(max(val, 0.0), 1.0)
        except ValueError:
            return 0.0
    return 0.0

# ==========================================
# System Prompts
# ==========================================
EXPRESSION_PROMPT = """
Task: Classify the facial expression in the image into exactly one of the following categories.

Allowed Categories:
1. happy (e.g., smiling, laughing, joyful)
2. surprise (e.g., raised eyebrows, open mouth, shocked)
3. confuse (e.g., frowning, puzzled, unsure)
4. neutral (e.g., blank face, calm, no strong emotion)
5. sad (e.g., crying, frowning mouth corners, gloomy)
6. others (e.g., angry, disgusted, fearful, or if the expression is unclear)

Constraints:
- You must ONLY output one word from the list above.
- Do NOT output any punctuation.
- If the expression is ambiguous, choose 'others'.
"""

def get_scenario_prompt(input_text):
    return f"""
Task: Scenario Consistency Check

Input Text: "{input_text}"

You need to perform a two-step analysis:
Step 1: Text Extraction (Mental Process)
Analyze the Input Text and extract the **"Unique Situational Descriptor"**. 
- IGNORE: Gender, Standard Pose, and Basic Emotion labels.
- TARGET: The specific *cause* of the emotion, the *environmental element*, or the *subtle physical detail*.

Step 2: Visual Verification
Look at the image. Does the visual content match the **"Unique Situational Descriptor"**?

Output Format:
- Extracted Context: ...
- Visual Evidence: ...
- Match Score: [0.0 to 1.0]

Constraints:
- 1.0: Specific scenario clearly visible.
- 0.5: General vibe matches, specific detail missing.
- 0.0: Scenario absent.
"""

# ==========================================
# 核心處理邏輯
# ==========================================
def process_task(task_name, swapped_dir, t2i_dir, json_path, pipe, gen_configs):
    gen_config_expr, gen_config_scen = gen_configs
    
    print(f"\n🔹 Processing Task: [{task_name}]")
    print(f"   📂 Swapped Dir: {swapped_dir}")
    print(f"   📂 T2I Source:  {t2i_dir}")
    
    if not os.path.exists(json_path):
        print(f"   ❌ JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 統計數據初始化
    stats = {
        'expr_correct_swap': 0, 'expr_total_swap': 0,
        'scen_score_swap': 0.0, 'scen_count_swap': 0,
        'expr_correct_t2i': 0, 'expr_total_t2i': 0,
        'scen_score_t2i': 0.0, 'scen_count_t2i': 0
    }

    # 批次處理設定
    batch_size = 4  # 顯存小可設為 2 或 4
    valid_extensions = ['.png', '.jpg', '.jpeg']

    for i in tqdm(range(0, len(data_list), batch_size), desc=f"   Running {task_name}"):
        batch_items = data_list[i : i + batch_size]
        
        # 輸入容器
        expr_inputs = [] 
        scen_inputs = []
        # 映射表：紀錄推論結果屬於哪個 Item 的哪個類型 (swap/t2i)
        map_indices = [] 

        for idx, item in enumerate(batch_items):
            prompt_text = item.get('prompt', '').strip()
            raw_filename = item.get('image', '').strip()
            
            # 初始化欄位 (避免找不到圖時報錯)
            item.setdefault('vlm_expression', "not_found")
            item.setdefault('scenario_score', 0.0)
            item.setdefault('vlm_expression_t2i', "not_found")
            item.setdefault('scenario_score_t2i', 0.0)
            
            # --- 1. 準備 Swapped Image ---
            swap_path, _ = try_find_file(swapped_dir, raw_filename, valid_extensions)
            if swap_path:
                try:
                    img_swap = load_image(swap_path)
                    
                    # 加入表情任務
                    expr_inputs.append((EXPRESSION_PROMPT, img_swap))
                    # 加入情境任務
                    scen_inputs.append((get_scenario_prompt(prompt_text), img_swap))
                    
                    map_indices.append({'local_idx': idx, 'type': 'swap'})
                except: pass

            # --- 2. 準備 T2I Original Image ---
            t2i_path, _ = find_target_by_prompt(t2i_dir, prompt_text)
            if t2i_path:
                try:
                    img_t2i = load_image(t2i_path)
                    
                    # 加入表情任務
                    expr_inputs.append((EXPRESSION_PROMPT, img_t2i))
                    # 加入情境任務
                    scen_inputs.append((get_scenario_prompt(prompt_text), img_t2i))
                    
                    map_indices.append({'local_idx': idx, 'type': 't2i'})
                except: pass

        if not map_indices: continue

        # --- 執行推論 (Inference) ---
        try:
            # 一次送入該 Batch 所有圖片 (Swap + T2I)
            expr_resps = pipe(expr_inputs, gen_config=gen_config_expr)
            scen_resps = pipe(scen_inputs, gen_config=gen_config_scen)
            
            # --- 將結果寫回對應 Item ---
            for meta, r_expr, r_scen in zip(map_indices, expr_resps, scen_resps):
                item = batch_items[meta['local_idx']]
                img_type = meta['type']
                
                # 解析表情
                pred_expr = r_expr.text.strip().lower().replace(".", "").replace("'", "")
                gt_expr = item.get('gt_expression', '').lower().strip()
                
                # 解析情境分數
                scen_text = r_scen.text
                scen_score = extract_score(scen_text)

                # 根據圖片類型存入不同欄位
                if img_type == 'swap':
                    item['vlm_expression'] = pred_expr
                    item['scenario_score'] = scen_score
                    item['scenario_reasoning'] = scen_text
                    
                    if gt_expr:
                        is_corr = (pred_expr == gt_expr)
                        item['expression_correct'] = 1 if is_corr else 0
                        stats['expr_total_swap'] += 1
                        if is_corr: stats['expr_correct_swap'] += 1
                    
                    if scen_score >= 0:
                        stats['scen_score_swap'] += scen_score
                        stats['scen_count_swap'] += 1

                elif img_type == 't2i':
                    item['vlm_expression_t2i'] = pred_expr
                    item['scenario_score_t2i'] = scen_score
                    item['scenario_reasoning_t2i'] = scen_text
                    
                    if gt_expr:
                        is_corr = (pred_expr == gt_expr)
                        item['expression_correct_t2i'] = 1 if is_corr else 0
                        stats['expr_total_t2i'] += 1
                        if is_corr: stats['expr_correct_t2i'] += 1
                    
                    if scen_score >= 0:
                        stats['scen_score_t2i'] += scen_score
                        stats['scen_count_t2i'] += 1

        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    # 存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    # 計算平均值
    def safe_div(a, b): return a/b if b > 0 else 0.0
    
    return {
        'name': task_name,
        'acc_expr_swap': safe_div(stats['expr_correct_swap'], stats['expr_total_swap']) * 100,
        'acc_expr_t2i': safe_div(stats['expr_correct_t2i'], stats['expr_total_t2i']) * 100,
        'avg_scen_swap': safe_div(stats['scen_score_swap'], stats['scen_count_swap']),
        'avg_scen_t2i': safe_div(stats['scen_score_t2i'], stats['scen_count_t2i'])
    }

# ==========================================
# 主程式
# ==========================================
if __name__ == '__main__':
    # 1. 載入模型 (一次性)
    print("🚀 Loading InternVL Model...")
    backend_config = PytorchEngineConfig(tp=1, session_len=4096, cache_max_entry_count=0.2)
    pipe = pipeline('OpenGVLab/InternVL3_5-8B', backend_config=backend_config)
    
    gen_config_expr = GenerationConfig(top_k=1, temperature=0.0)
    gen_config_scen = GenerationConfig(top_k=1, temperature=0.1)
    gen_configs = (gen_config_expr, gen_config_scen)

    # 2. 設定區域
    DEFAULT_T2I_DIR = './pixart_outputs'
    SOURCE_JSON = 'gt.json'

    METHOD_DIRS = {
        'PixArt': './faceswap_results/pixart',
        # 'Janus': './faceswap_results/janus',
        # 'Infinity': './faceswap_results/infinity',
        # 'ShowO2': './faceswap_results/showo2'
    }

    results_summary = []
    print(f"\n📋 Starting Batch VLM Evaluation (Source: {SOURCE_JSON})...")

    for name, swapped_dir in METHOD_DIRS.items():
        if not os.path.exists(swapped_dir):
            print(f"⚠️ Skipping {name}: Directory not found")
            continue

        res = process_task(name, swapped_dir, DEFAULT_T2I_DIR, SOURCE_JSON, pipe, gen_configs)
        if res: results_summary.append(res)

    # 最終比較總表
    print("\n" + "="*95)
    print(f"{'Method':<10} | {'Expr Acc (Swap)':<15} | {'Expr Acc (T2I)':<15} | {'Scen Score (Swap)':<17} | {'Scen Score (T2I)':<17}")
    print("-" * 95)
    for res in results_summary:
        print(f"{res['name']:<10} | {res['acc_expr_swap']:<15.2f}% | {res['acc_expr_t2i']:<15.2f}% | {res['avg_scen_swap']:<17.4f} | {res['avg_scen_t2i']:<17.4f}")
    print("="*95)
    print("✅ All tasks completed.")