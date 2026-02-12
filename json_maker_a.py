import os
import json
import argparse

def generate_json_from_folder(folder_path, output_file):
    # 支援的圖片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    json_data = []
    
    # 1. 取得資料夾內所有檔案
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        # 2. 排序優化：
        # 排序時我們暫時轉成 int，這樣 10 才不會排在 2 前面。
        # 但這不會改變原始檔名，我們稍後還是拿得到 "00051"
        try:
            files.sort(key=lambda f: int(f.split('_')[0]))
        except ValueError:
            # 如果檔名開頭不是數字，就退回一般排序
            files.sort()
            
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 '{folder_path}'")
        return

    for filename in files:
        # 去除副檔名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 3. 解析 {index}_{prompt} 格式
        if '_' in name_without_ext:
            parts = name_without_ext.split('_', 1)
            
            # 【關鍵修改】：直接保留原始字串，不轉 int
            index_str = parts[0]  # 例如 "00051"
            prompt_part = parts[1]
            
        else:
            print(f"警告: 檔案 '{filename}' 不符合 {{index}}_{{prompt}} 格式，將使用原始檔名。")
            index_str = "0" # 或其他預設值
            prompt_part = name_without_ext

        # 4. 建立資料結構
        entry = {
            "id": index_str,  # 這裡是字串，會保留 "00051"
            "image": filename,
            "prompt": prompt_part
        }
        json_data.append(entry)

    # 確保輸出路徑存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. 儲存為 JSON 檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"成功！已處理 {len(json_data)} 張圖片。")
    print(f"範例 ID 格式確認: {json_data[0]['id'] if json_data else '無資料'}")
    print(f"結果儲存於: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate JSON metadata preserving ID format (e.g., 00051)")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    generate_json_from_folder(args.folder, args.output)