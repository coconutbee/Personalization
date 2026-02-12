import argparse
import os
import json
import re
import torch
from tqdm import tqdm
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# ==========================================
# 1. 檔案搜尋工具
# ==========================================
def smart_find_swapped_image(base_dir, prompt):
    """根據 prompt 找尋 swapped 資料夾中的圖片"""
    if not base_dir or not os.path.exists(base_dir): return None
    for filename in os.listdir(base_dir):
        if "_" in filename:
            parts = filename.split('_', 1) 
            name_part = os.path.splitext(parts[1])[0]
            if name_part.strip() == prompt.strip():
                return os.path.join(base_dir, filename)
    return None

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 找尋對應的 T2I 原圖 (支援 00051_prompt.jpg 格式)"""
    if not prompt or not base_dir or not os.path.exists(base_dir): return None
    
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip().lower()
    
    target_norm = normalize(prompt)
    
    for filename in os.listdir(base_dir):
        # 取得不含副檔名的檔名
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. 嘗試直接匹配 (例如 prompt.jpg)
        if normalize(name_no_ext) == target_norm:
            return os.path.join(base_dir, filename)
            
        # 2. 嘗試忽略 ID 前綴匹配 (例如 00051_prompt.jpg)
        if "_" in name_no_ext:
            # 只切分第一個底線
            parts = name_no_ext.split('_', 1)
            # 比較後半段是否與 prompt 相同
            if len(parts) > 1 and normalize(parts[1]) == target_norm:
                return os.path.join(base_dir, filename)
                
    return None

# ==========================================
# 2. 工具函式：提取分數
# ==========================================
def extract_score(text: str) -> float:
    match = re.search(r"(?:Match )?Score[:\s\n*]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return min(max(val, 0.0), 1.0)
        except ValueError:
            return 0.0
    return 0.0

# ==========================================
# 3. System Prompts
# ==========================================
EXPRESSION_PROMPT = """
Task: Classify the facial expression in the image into exactly one of the following categories.
Allowed Categories:
1. happy
2. surprise
3. confuse
4. neutral
5. sad
6. others
Constraints: Output ONLY one word.
"""

def get_scenario_prompt(input_text):
    return f"""
Task: Scenario Consistency Check
Input Text: "{input_text}"
Step 1: Text Extraction. Extract the "Unique Situational Descriptor" (cause, environment, detail).
Step 2: Visual Verification. Does the image match?
Output Format:
- Extracted Context: ...
- Visual Evidence: ...
- Match Score: [0.0 to 1.0]
"""

# ==========================================
# 4. 核心處理邏輯 (含 Cache 與 Mode 機制)
# ==========================================
def process_task(task_name, swapped_dir, t2i_dir, json_path, pipe, gen_configs, mode='full'):
    """
    mode: 'full' (預設，跑 swap + t2i) 或 't2i' (只跑 t2i)
    """
    gen_config_expr, gen_config_scen = gen_configs
    
    print(f"\n🔹 Processing Task: [{task_name}] | Mode: {mode.upper()}")
    if mode == 'full':
        print(f"   📂 Swapped Dir: {swapped_dir}")
    print(f"   📂 T2I Source:  {t2i_dir}")
    
    if not os.path.exists(json_path):
        print(f"   ❌ JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    stats = {
        'expr_correct_swap': 0, 'expr_total_swap': 0,
        'scen_score_swap': 0.0, 'scen_count_swap': 0,
        'expr_correct_t2i': 0, 'expr_total_t2i': 0,
        'scen_score_t2i': 0.0, 'scen_count_t2i': 0,
        'unique_t2i_processed': 0  # 追蹤實際跑了幾張 T2I
    }

    # --- Cache 機制 ---
    # Key: prompt (因為 prompt 決定了 T2I 圖片)
    t2i_cache = {}

    batch_size = 4
    
    for i in tqdm(range(0, len(data_list), batch_size), desc=f"   Running {task_name}"):
        batch_items = data_list[i : i + batch_size]
        
        expr_inputs = [] 
        scen_inputs = []
        map_indices = [] 
        
        # 暫存需要從 Cache 填值的項目
        cache_fill_queue = []

        for idx, item in enumerate(batch_items):
            prompt_text = item.get('prompt', '').strip()
            
            # 初始化欄位 (根據 mode 不同，有些可能用不到但初始化無妨)
            if mode == 'full':
                item.setdefault('swap_vlm_expression', "not_found")
                item.setdefault('swap_scenario_score', 0.0)
            
            item.setdefault('t2i_vlm_expression', "not_found")
            item.setdefault('t2i_scenario_score', 0.0)

            # --- 1. Swapped Image (僅在 full 模式下執行) ---
            if mode == 'full':
                swap_path = smart_find_swapped_image(swapped_dir, prompt_text)
                if swap_path:
                    try:
                        img_swap = load_image(swap_path)
                        expr_inputs.append((EXPRESSION_PROMPT, img_swap))
                        scen_inputs.append((get_scenario_prompt(prompt_text), img_swap))
                        map_indices.append({'local_idx': idx, 'type': 'swap'})
                    except Exception as e:
                        print(f"Error loading swap image: {e}")

            # --- 2. T2I Original Image (檢查 Cache) ---
            # 無論是 full 還是 t2i 模式，都要跑這一段
            if prompt_text in t2i_cache:
                # 命中 Cache！放入待填隊列，稍後直接複製
                cache_fill_queue.append({'local_idx': idx, 'prompt': prompt_text})
            else:
                # 未命中，需要跑 VLM
                t2i_path = find_target_by_prompt(t2i_dir, prompt_text)
                if t2i_path:
                    try:
                        img_t2i = load_image(t2i_path)
                        expr_inputs.append((EXPRESSION_PROMPT, img_t2i))
                        scen_inputs.append((get_scenario_prompt(prompt_text), img_t2i))
                        # 標記為 't2i_new'，表示跑完要更新 Cache
                        map_indices.append({'local_idx': idx, 'type': 't2i_new', 'prompt': prompt_text})
                    except Exception as e:
                        print(f"Error loading t2i image: {e}")

        # --- 3. 執行推論 (Inference) ---
        if expr_inputs:
            try:
                expr_resps = pipe(expr_inputs, gen_config=gen_config_expr)
                scen_resps = pipe(scen_inputs, gen_config=gen_config_scen)
                
                # 寫回結果
                for meta, r_expr, r_scen in zip(map_indices, expr_resps, scen_resps):
                    current_item = batch_items[meta['local_idx']]
                    img_type = meta['type']
                    
                    pred_expr = r_expr.text.strip().lower().replace(".", "").replace("'", "")
                    scen_text = r_scen.text
                    scen_score = extract_score(scen_text)
                    gt_expr = current_item.get('gt_expression', '').lower().strip()

                    # 寫入 Swap 結果
                    if img_type == 'swap':
                        current_item['swap_vlm_expression'] = pred_expr
                        current_item['swap_scenario_score'] = scen_score
                        current_item['swap_scenario_reasoning'] = scen_text
                        
                        if gt_expr:
                            is_corr = (pred_expr == gt_expr)
                            current_item['expression_correct'] = 1 if is_corr else 0
                            stats['expr_total_swap'] += 1
                            if is_corr: stats['expr_correct_swap'] += 1
                        if scen_score >= 0:
                            stats['scen_score_swap'] += scen_score
                            stats['scen_count_swap'] += 1

                    # 寫入新的 T2I 結果並更新 Cache
                    elif img_type == 't2i_new':
                        is_corr = (pred_expr == gt_expr) if gt_expr else False
                        
                        # 構建結果物件
                        t2i_result = {
                            't2i_vlm_expression': pred_expr,
                            't2i_scenario_score': scen_score,
                            't2i_scenario_reasoning': scen_text,
                            'expression_correct_t2i': 1 if is_corr else 0
                        }
                        
                        # 更新當前 Item
                        current_item.update(t2i_result)
                        
                        # 更新 Cache
                        t2i_cache[meta['prompt']] = t2i_result
                        stats['unique_t2i_processed'] += 1

            except Exception as e:
                print(f"Error in VLM inference: {e}")

        # --- 4. 應用 Cache (填補重複的 T2I) ---
        for meta in cache_fill_queue:
            current_item = batch_items[meta['local_idx']]
            prompt = meta['prompt']
            if prompt in t2i_cache:
                current_item.update(t2i_cache[prompt])

        # --- 5. 統計累加 (T2I 部分) ---
        for item in batch_items:
            # 確保有資料才統計
            if item.get('t2i_vlm_expression', 'not_found') != "not_found":
                stats['expr_total_t2i'] += 1
                if item.get('expression_correct_t2i', 0) == 1:
                    stats['expr_correct_t2i'] += 1
                
                stats['scen_score_t2i'] += item.get('t2i_scenario_score', 0)
                stats['scen_count_t2i'] += 1

    # --- 6. 最終完整性檢查 (Sanity Check) ---
    print("\n🔍 Verifying Data Integrity...")
    missing_count = 0
    for i, item in enumerate(data_list):
        keys_to_check = ['t2i_vlm_expression']
        if mode == 'full':
            keys_to_check.append('swap_vlm_expression')
            
        if any(item.get(k) == "not_found" for k in keys_to_check):
            missing_count += 1
    
    if missing_count == 0:
        print(f"✅ All {len(data_list)} items have been evaluated successfully!")
    else:
        print(f"⚠️ Warning: {missing_count}/{len(data_list)} items are missing results.")
    
    print(f"ℹ️  Actual Unique T2I Images Processed by VLM: {stats['unique_t2i_processed']}")

    # 存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    def safe_div(a, b): return a/b if b > 0 else 0.0
    
    # 回傳統計結果
    result_dict = {
        'name': task_name,
        'mode': mode,
        'acc_expr_t2i': safe_div(stats['expr_correct_t2i'], stats['expr_total_t2i']) * 100,
        'avg_scen_t2i': safe_div(stats['scen_score_t2i'], stats['scen_count_t2i']),
        'count_t2i': stats['scen_count_t2i']
    }

    if mode == 'full':
        result_dict.update({
            'acc_expr_swap': safe_div(stats['expr_correct_swap'], stats['expr_total_swap']) * 100,
            'avg_scen_swap': safe_div(stats['scen_score_swap'], stats['scen_count_swap']),
            'count_swap': stats['scen_count_swap']
        })
    
    return result_dict

# ==========================================
# 5. 主程式
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch VLM Evaluation (With T2I Caching)")
    parser.add_argument("--json", type=str, default="metadata.json", help="Source JSON file")
    parser.add_argument("--name", type=str, default="pixart", help="Task Name")
    parser.add_argument("--t2i", type=str, required=True, help="T2I Dir (Required)")
    # 將 swap 改為非必填 (default None)，因為 t2i 模式不需要
    parser.add_argument("--swap", type=str, default=None, help="Swapped Dir (Optional if mode=t2i)")
    parser.add_argument("--ref", type=str, default="", help="Not used")
    # 新增 mode 參數
    parser.add_argument("--mode", type=str, default="full", choices=["full", "t2i"], help="Evaluation Mode: 'full' (Swap+T2I) or 't2i' (T2I Only)")
    
    args = parser.parse_args()

    # 參數檢查
    if args.mode == 'full' and not args.swap:
        parser.error("--swap argument is required when mode is 'full'")

    print("🚀 Loading InternVL Model...")
    backend_config = PytorchEngineConfig(tp=1, session_len=4096, cache_max_entry_count=0.2)
    pipe = pipeline('OpenGVLab/InternVL3_5-8B', backend_config=backend_config)
    
    gen_config_expr = GenerationConfig(top_k=1, temperature=0.0)
    gen_config_scen = GenerationConfig(top_k=1, temperature=0.1)
    gen_configs = (gen_config_expr, gen_config_scen)

    print(f"\n📋 Starting Batch VLM Evaluation (Source: {args.json})...")
    res = process_task(args.name, args.swap, args.t2i, args.json, pipe, gen_configs, mode=args.mode)
    
    if res:
        print("\n" + "="*95)
        if args.mode == 'full':
            print(f"{'Method':<10} | {'Expr Acc (Swap)':<17} | {'Expr Acc (T2I)':<17} | {'Scen Score (Swap)':<19} | {'Scen Score (T2I)':<19}")
            print("-" * 95)
            print(f"{res['name']:<10} | {res['acc_expr_swap']:<16.2f}% | {res['acc_expr_t2i']:<16.2f}% | {res['avg_scen_swap']:<19.4f} | {res['avg_scen_t2i']:<19.4f}")
        else:
            # T2I Only 的輸出格式
            print(f"{'Method':<10} | {'Expr Acc (T2I)':<20} | {'Scen Score (T2I)':<20}")
            print("-" * 60)
            print(f"{res['name']:<10} | {res['acc_expr_t2i']:<19.2f}% | {res['avg_scen_t2i']:<19.4f}")
        print("="*95)
        print("✅ Process Completed.")