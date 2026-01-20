import os
import csv
from tqdm import tqdm
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

def run_scenario_check():
    # --- 模型配置 ---
    model_path = 'OpenGVLab/InternVL3_5-8B'
    # 針對顯存優化：session_len 縮短，cache 分配減少
    backend_config = PytorchEngineConfig(tp=1, session_len=8192, cache_max_entry_count=0.2)
    pipe = pipeline(model_path, backend_config=backend_config)
    
    # --- 路徑配置 ---
    # base_folder_path = '/home/ee303/.cache/kagglehub/datasets/dollyprajapati182/balanced-affectnet/versions/1/test'
    # base_folder_path = '/home/ee303/.cache/kagglehub/datasets/dollyprajapati182/balanced-caer-s-dataset-7575-grayscale/versions/1/test'
    # base_folder_path = '/media/ee303/disk1/Infinity/output_images'
    # base_folder_path = '/media/ee303/disk1/Janus/generated_samples'
    # base_folder_path = '/media/ee303/disk1/Show-o/show-o2/outputs/generated_images_432'
    base_folder_path = '/media/ee303/disk2/style_generation/diffusers/pose_output256'
    output_csv = 'internvl_open_exps_pixart.csv'
    batch_size = 1  # 顯存充足可調回 16 或 32

    # --- 1. 蒐集檔案與標籤 ---
    all_data = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG')
    
    print("正在掃描檔案...")
    for root, dirs, files in os.walk(base_folder_path):
        for file_name in files:
            if file_name.endswith(valid_extensions):
                full_path = os.path.join(root, file_name)
                # 子資料夾名稱即為 Ground Truth
                label = os.path.basename(root) 
                all_data.append((full_path, file_name, label))

    total_images = len(all_data)
    print(f"總共找到的圖片數量: {total_images}")
    
    if total_images == 0:
        print("錯誤：找不到任何匹配的圖片檔案，請檢查路徑！")
        return

    # --- 2. 初始化統計變數 ---
    correct_count = 0
    total_processed = 0

    # --- 3. 開始批次處理與評估 ---
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 增加一欄 is_match 方便後續手動檢查
        writer.writerow(['path', 'expression_internvl'])

        for i in tqdm(range(0, total_images, batch_size), desc="Batch Processing"):
            batch_data = all_data[i : i + batch_size]
            prompts = []
            current_batch_info = [] # 儲存當前 batch 的 metadata
            
            for full_path, file_name, gt in batch_data:
                try:
                    img = load_image(full_path)
                    prompts.append((
                        """
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
                        - Do NOT output any punctuation, explanation, or extra text.
                        - If the expression is ambiguous or fits multiple categories not listed (like anger), choose 'others'.

                        Output Example:
                        happy
                        """,
                        img
                    ))
                    current_batch_info.append((file_name, gt))
                except Exception as e:
                    print(f"\n跳過損壞圖片 {file_name}: {e}")
                    continue

            if not prompts:
                continue

            try:
                # 執行推理
                gen_config = GenerationConfig(top_k=1, temperature=0.0)
                responses = pipe(prompts, gen_config=gen_config)

                for response, info in zip(responses, current_batch_info):
                    file_name, ground_truth = info
                    # 取得預測結果並清理格式
                    prediction = response.text.strip().lower().replace("'", "").replace(".", "")
                    
                    # 比對正確率 (不分大小寫)
                    # is_match = 1 if prediction == ground_truth.lower() else 0
                    # if is_match:
                    #     correct_count += 1
                    # total_processed += 1
                    
                    # 寫入 CSV
                    writer.writerow([file_name, prediction])
                
                f.flush()
            except Exception as e:
                print(f"\nBatch 處理失敗: {e}")
                continue

    # --- 4. 輸出最終結果 ---
    # accuracy = correct_count / total_processed if total_processed > 0 else 0
    print("\n" + "="*40)
    print(f"評估完成！")
    print(f"總處理數: {total_processed}")
    # print(f"正確數: {correct_count}")
    # print(f"整體正確率 (Accuracy): {accuracy:.2%}")
    print(f"結果已儲存至: {output_csv}")
    print("="*40)

if __name__ == '__main__':
    run_scenario_check()