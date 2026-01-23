import argparse
import os
import json
import re
import torch
from tqdm import tqdm
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# ==========================================
# 工具函式：提取分數
# ==========================================
def extract_score(text: str) -> float:
    """
    從 VLM 回覆中提取分數 (支援 0.9, 1.0, 1 等格式)
    若失敗回傳 0.0
    """
    match = re.search(r"(?:Match )?Score[:\s\n*]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            # 確保分數在 0.0 到 1.0 之間
            return min(max(val, 0.0), 1.0)
        except ValueError:
            return 0.0
    return 0.0

# ==========================================
# System Prompts
# ==========================================
# 1. 表情分類 Prompt
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

# 2. 情境分析 Prompt 模板
def get_scenario_prompt(input_text):
    return f"""
Task: Scenario Consistency Check

Input Text: "{input_text}"

You need to perform a two-step analysis:

Step 1: Text Extraction (Mental Process)
Analyze the Input Text and extract the **"Unique Situational Descriptor"**. 
- IGNORE: Gender (boy, girl), Standard Pose (turns head, looks up), and Basic Emotion labels (happy, sad).
- TARGET: The specific *cause* of the emotion, the *environmental element*, or the *subtle physical detail*.
- Examples:
  - "A girl faces downward with a shy smile, cheeks slightly blushing" -> Target: "cheeks slightly blushing"
  - "A boy looks upward... as snowflakes fall on his face" -> Target: "snowflakes fall on his face"

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
# 主程式
# ==========================================
def run_merged_eval(method, base_folder_path, json_path):
    # 1. 模型配置
    print("🚀 正在載入 InternVL 模型...")
    # 若顯存不足，可調低 session_len (例如 2048)
    backend_config = PytorchEngineConfig(tp=1, session_len=4096, cache_max_entry_count=0.2)
    pipe = pipeline('OpenGVLab/InternVL3_5-8B', backend_config=backend_config)
    
    # 設定生成參數
    gen_config_expr = GenerationConfig(top_k=1, temperature=0.0) # 表情需要精準
    gen_config_scen = GenerationConfig(top_k=1, temperature=0.1) # 情境允許微量創意以解析語意

    # 2. 讀取 JSON
    print(f"📂 正在讀取 JSON: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ 錯誤: 找不到檔案 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"📊 總共需處理: {len(data_list)} 筆資料")

    # 3. 批次推論迴圈
    batch_size = 4  # 依顯存調整 (建議 4-8)
    
    # 統計用
    expr_correct_count = 0
    expr_total_valid = 0
    total_scenario_score = 0.0
    scenario_count = 0

    for i in tqdm(range(0, len(data_list), batch_size), desc="VLM Evaluating"):
        batch_items = data_list[i : i + batch_size]
        
        # 準備容器
        expr_inputs = []   # 表情推論用 [(prompt, img), ...]
        scen_inputs = []   # 情境推論用 [(prompt, img), ...]
        valid_indices = [] # 紀錄這批裡面哪些是有效讀取圖片的 (對應 batch_items 的 index)

        for idx, item in enumerate(batch_items):
            prompt_text = item.get('prompt', '').strip()
            
            # 取得原始檔名 (例如 "1.jpg")
            raw_filename = item.get('image', '').strip()

            if not raw_filename:
                print(f"[Warning] ID {item.get('id')} 缺少 image 欄位，跳過")
                continue
            
            # === [關鍵修改] 加上 0_ 前綴 ===
            # 例如: "1.jpg" -> "0_1.jpg"
            image_filename = f"0_{raw_filename}"
            image_filename = image_filename.replace('jpg', 'png')            
            # 組裝完整路徑
            full_path = os.path.join(base_folder_path, image_filename)
            
            if not os.path.exists(full_path):
                # 檔案不存在的處理
                item['vlm_expression'] = "image_not_found"
                item['expression_correct'] = 0
                item['scenario_score'] = 0.0
                item['scenario_reasoning'] = f"Image file not found: {image_filename}"
                # print(f"[Warning] 找不到圖片: {full_path}")
                continue

            try:
                # 載入圖片 (lmdeploy 格式)
                img = load_image(full_path)
                
                # --- B. 準備兩個任務的 Prompt ---
                # 任務 1: 表情分類
                expr_inputs.append((EXPRESSION_PROMPT, img))
                
                # 任務 2: 情境分析 (動態生成 Prompt)
                scen_prompt = get_scenario_prompt(prompt_text)
                scen_inputs.append((scen_prompt, img))
                
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"\n[Error] 讀取失敗: {image_filename} | {e}")
                continue

        if not valid_indices:
            continue

        try:
            # --- C. 執行推論 (分兩次跑，但模型不用重載) ---
            
            # 1. 跑表情分類
            expr_responses = pipe(expr_inputs, gen_config=gen_config_expr)
            
            # 2. 跑情境分析
            scen_responses = pipe(scen_inputs, gen_config=gen_config_scen)

            # --- D. 處理結果並寫回 JSON ---
            for local_idx, resp_expr, resp_scen in zip(valid_indices, expr_responses, scen_responses):
                item = batch_items[local_idx]
                
                # [處理表情結果]
                pred_expr = resp_expr.text.strip().lower().replace(".", "").replace("'", "")
                item['vlm_expression'] = pred_expr
                
                gt_expr = item.get('gt_expression', '').lower().strip()
                if gt_expr:
                    is_correct = (pred_expr == gt_expr)
                    # 轉成 0 或 1
                    item['expression_correct'] = 1 if is_correct else 0
                    
                    expr_total_valid += 1
                    if is_correct: expr_correct_count += 1
                else:
                    item['expression_correct'] = None

                # [處理情境結果]
                scen_text = resp_scen.text
                score = extract_score(scen_text)
                
                item['scenario_reasoning'] = scen_text
                item['scenario_score'] = score
                
                if score >= 0:
                    total_scenario_score += score
                    scenario_count += 1

        except Exception as e:
            print(f"\n[Fatal Error] 推理中斷: {e}")
            break

    # 4. 最終存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 完成！結果已更新至 {json_path}")
    
    # 顯示統計數據
    if expr_total_valid > 0:
        acc = (expr_correct_count / expr_total_valid) * 100
        print(f"😐 表情準確率: {acc:.2f}% ({expr_correct_count}/{expr_total_valid})")
    
    if scenario_count > 0:
        avg_scen = total_scenario_score / scenario_count
        print(f"🎬 平均情境分數: {avg_scen:.2f} (共 {scenario_count} 筆有效評分)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='pixart', help="folder selection")
    parser.add_argument("--json", type=str, default='gt.json', help="Path to the JSON file")
    args = parser.parse_args()

    # 路徑映射 (已更新為 faceswap_results)
    path_map = {
        'pixart': './faceswap_results/pixart',
        'janus': './faceswap_results/janus',
        'infinity': './faceswap_results/infinity',
        'showo2': './faceswap_results/showo2'
    }
    
    # 預設路徑
    base_folder_path = path_map.get(args.method, './output')

    print(f"Method: {args.method}")
    print(f"Image Folder: {base_folder_path}")

    run_merged_eval(args.method, base_folder_path, args.json)