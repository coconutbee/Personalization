import os
import json
import argparse

def generate_json_from_folder(folder_path, output_file):
    # 支援的圖片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    json_data = []
    
    # 取得資料夾內所有檔案並排序
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 '{folder_path}'")
        return

    for index, filename in enumerate(files):
        # 提取不含副檔名的檔名作為 prompt (或者你可以直接保留完整檔名)
        prompt_name = os.path.splitext(filename)[0]
        
        # 建立資料結構
        entry = {
            "id": index,
            "image": filename,
            "prompt": prompt_name
        }
        json_data.append(entry)

    # 儲存為 JSON 檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"成功！已處理 {len(json_data)} 張圖片，結果儲存於: {output_file}")



if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Generate JSON metadata from image folder")
    args.add_argument('--folder', type=str, required=True, help='Path to the folder containing images')
    args.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parsed_args = args.parse_args()
    target_folder = parsed_args.folder
    output_json = parsed_args.output

generate_json_from_folder(target_folder, output_json)