import os
import torch
import json
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer
from omegaconf import OmegaConf

from lavis.common.registry import registry
from lavis.processors import load_processor

# ==========================================
# 0. Patch (防止 import error) - 維持原樣
# ==========================================
def patch_lavis_library():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "lavis", "models", "__init__.py"),
        os.path.join(os.path.dirname(base_dir), "EvalMuse", "lavis", "models", "__init__.py"),
        os.path.join(base_dir, "EvalMuse", "lavis", "models", "__init__.py")
    ]
    target_file = None
    for p in possible_paths:
        if os.path.exists(p):
            target_file = p
            break
    if target_file:
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            if "from lavis.processors import load_preprocess" in content:
                new_content = content.replace("load_preprocess", "load_processor")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
        except Exception:
            pass
patch_lavis_library()

# ==========================================
# 1. 模型初始化
# ==========================================
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def load_fga_model(model_path):
    print(f"🚀 Loading FGA Model from {model_path}...")
    
    # 手動 Config (維持 364 解析度)
    manual_config = {
        "model": {
            "vit_model": "eva_clip_g",
            "img_size": 364,
            "image_size": 364,
            "drop_path_rate": 0,
            "use_grad_checkpoint": False,
            "vit_precision": "fp16",
            "freeze_vit": True,
            "num_query_token": 32,
            "cross_attention_freq": 2,
            "embed_dim": 256,
            "load_finetuned": False,
            "load_pretrained": False,
            "pretrained": None,
            "finetuned": None,
        }
    }
    cfg = OmegaConf.create(manual_config)
    model_cls = registry.get_model_class("fga_blip2")
    model = model_cls.from_config(cfg.model)
    
    # Tokenizer 設定
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    target_vocab_size = len(tokenizer)

    # Resize Embeddings
    if hasattr(model, "Qformer"):
        model.Qformer.bert.resize_token_embeddings(target_vocab_size)
        if hasattr(model.Qformer.cls, "predictions"):
            old_bias = model.Qformer.cls.predictions.bias
            new_bias = torch.nn.Parameter(torch.zeros(target_vocab_size))
            new_bias.data[:old_bias.shape[0]] = old_bias.data
            model.Qformer.cls.predictions.bias = new_bias
            
            old_decoder = model.Qformer.cls.predictions.decoder
            new_decoder = torch.nn.Linear(old_decoder.in_features, target_vocab_size)
            new_decoder.weight.data[:old_decoder.out_features, :] = old_decoder.weight.data
            new_decoder.bias.data[:old_decoder.out_features] = old_decoder.bias.data
            model.Qformer.cls.predictions.decoder = new_decoder
    
    # 載入權重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ Warning: Model path {model_path} not found!")

    model.eval()
    return model.to(device), tokenizer

def load_processors():
    vis_cfg = OmegaConf.create({"name": "blip_image_eval", "image_size": 364})
    text_cfg = OmegaConf.create({"name": "blip_caption"})
    vis_processor = load_processor("blip_image_eval", cfg=vis_cfg)
    text_processor = load_processor("blip_caption", cfg=text_cfg)
    return vis_processor, text_processor

# ==========================================
# 2. 核心計算函式
# ==========================================
def get_fga_metrics(model, image_tensor, clean_prompt, tokenizer, prompt_ids):
    """
    輸入處理好的 image_tensor 與 prompt，回傳 (總分, Token分數列表)
    """
    try:
        with torch.no_grad():
            alignment_score_output, scores = model.element_score(image_tensor.unsqueeze(0), [clean_prompt])

        # A. 處理整體分數
        final_score = 0.0
        if isinstance(alignment_score_output, dict):
            if 'score' in alignment_score_output: val = alignment_score_output['score']
            elif 'overall_score' in alignment_score_output: val = alignment_score_output['overall_score']
            elif 'itm_score' in alignment_score_output: val = alignment_score_output['itm_score']
            else: val = list(alignment_score_output.values())[0]
            final_score = val.item() if hasattr(val, 'item') else float(val)
        elif hasattr(alignment_score_output, 'item'):
            final_score = alignment_score_output.item()
        else:
            final_score = float(alignment_score_output)

        # B. 處理 Token 分數
        scores_tensor = scores.squeeze()
        if scores_tensor.dim() == 0: # Handle scalar tensor edge case
             scores_tensor = scores_tensor.unsqueeze(0)
             
        min_len = min(len(prompt_ids), len(scores_tensor))
        
        word_scores = []
        for idx in range(min_len):
            token_id = prompt_ids[idx]
            token_str = tokenizer.decode([token_id])
            token_score = scores_tensor[idx].item()
            
            if token_str in ['[CLS]', '[SEP]', '[DEC]']: continue
            word_scores.append([token_str, round(token_score, 4)]) # 保留4位小數
            
        return round(final_score, 4), word_scores

    except Exception as e:
        print(f"Error in FGA calculation: {e}")
        return None, None

# ==========================================
# 3. 檔案搜尋工具 (與 AdaFace/VLM 一致)
# ==========================================
def smart_find_swapped_image(base_dir, prompt):
    """
    根據 prompt 找尋 swapped 資料夾中的圖片。
    格式預設為: "隨機數_Prompt.jpg"
    """
    if not os.path.exists(base_dir): return None
    
    # 遍歷 swapped 資料夾
    for filename in os.listdir(base_dir):
        if "_" in filename:
            parts = filename.split('_', 1) 
            # rand_id = parts[0]
            name_part = os.path.splitext(parts[1])[0]
            
            # 比對 Prompt 是否一致
            if name_part.strip() == prompt.strip():
                return os.path.join(base_dir, filename)
    return None

def find_target_by_prompt(base_dir, prompt):
    """根據 Prompt 找尋對應的 T2I 原圖"""
    if not prompt or not os.path.exists(base_dir): return None
    
    def normalize(text):
        return text.replace("’", "'").replace("‘", "'").strip()
    
    target_norm = normalize(prompt)
    for filename in os.listdir(base_dir):
        if normalize(os.path.splitext(filename)[0]) == target_norm:
            return os.path.join(base_dir, filename)
    return None

# ==========================================
# 4. 單一任務處理邏輯
# ==========================================
def process_task(task_name, swapped_dir, t2i_dir, json_path, components):
    model, tokenizer, vis_proc, text_proc = components
    
    print(f"\n🔹 Processing Task: [{task_name}]")
    print(f"   📂 Swapped Dir: {swapped_dir}")
    print(f"   📂 T2I Source:  {t2i_dir}")

    if not os.path.exists(json_path):
        print(f"   ❌ JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {'orig': [], 'swap': []}

    for item in tqdm(data, desc=f"   Running {task_name}"):
        prompt = item.get('prompt', '').strip()
        
        # 預處理 Text
        clean_prompt = text_proc(prompt)
        prompt_ids = tokenizer(clean_prompt).input_ids

        # 初始化欄位
        item.setdefault('fga_orig_score', None)
        item.setdefault('fga_orig_tokens', None)
        item.setdefault('fga_swap_score', None)
        item.setdefault('fga_swap_tokens', None)

        # --- A. 計算 Swapped Image 分數 ---
        swapped_path = smart_find_swapped_image(swapped_dir, prompt)
        if swapped_path:
            try:
                raw_img = Image.open(swapped_path).convert("RGB")
                img_tensor = vis_proc(raw_img).to(device)
                
                score, tokens = get_fga_metrics(model, img_tensor, clean_prompt, tokenizer, prompt_ids)
                
                item['fga_swap_score'] = score
                item['fga_swap_tokens'] = tokens
                if score is not None: stats['swap'].append(score)
            except Exception as e:
                print(f"Error reading swapped image {swapped_path}: {e}")

        # --- B. 計算 Original T2I Image 分數 ---
        t2i_path = find_target_by_prompt(t2i_dir, prompt)
        if t2i_path:
            try:
                raw_img = Image.open(t2i_path).convert("RGB")
                img_tensor = vis_proc(raw_img).to(device)
                
                score, tokens = get_fga_metrics(model, img_tensor, clean_prompt, tokenizer, prompt_ids)
                
                item['fga_orig_score'] = score
                item['fga_orig_tokens'] = tokens
                if score is not None: stats['orig'].append(score)
            except Exception as e:
                pass # T2I 不存在就算了

    # 存檔
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 計算平均
    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    return {
        'name': task_name,
        'avg_orig': avg(stats['orig']),
        'avg_swap': avg(stats['swap']),
        'count_orig': len(stats['orig']),
        'count_swap': len(stats['swap'])
    }

# ==========================================
# 5. 主程式
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch FGA Evaluation")
    parser.add_argument('--model_path', type=str, default='./EvalMuse/fga_blip2.pth')
    parser.add_argument("--json", type=str, default="metadata.json", help="Source JSON file for evaluation")
    parser.add_argument("--name", type=str, default="pixart", help="Name of the method/task")
    parser.add_argument("--t2i", type=str, default="/media/ee303/disk2/style_generation/diffusers/pixart_test", help="Directory of T2I original images")
    parser.add_argument("--swap", type=str, default="/media/ee303/disk2/JACK/FACE_SWAPED_pixart_test", help="Directory of swapped images")
    args = parser.parse_args()

    # 1. 載入模型 (一次性)
    model, tokenizer = load_fga_model(args.model_path)
    vis_processor, text_processor = load_processors()
    components = (model, tokenizer, vis_processor, text_processor)

    summary = []
    print(f"\n📋 Starting Batch FGA Evaluation (Source: {args.json})...")

    # 執行任務
    if not os.path.exists(args.swap):
        print(f"⚠️ Skipping {args.name}: Directory not found ({args.swap})")
    else:
        res = process_task(
            args.name, 
            args.swap, 
            args.t2i, 
            args.json, 
            components
        )
        if res: summary.append(res)

    # 3. 輸出總表
    print("\n" + "="*80)
    print(f"{'Method':<15} | {'FGA Orig (T2I)':<18} | {'FGA Swapped':<18} | {'Count':<8}")
    print("-" * 80)
    for res in summary:
        print(f"{res['name']:<15} | {res['avg_orig']:<18.4f} | {res['avg_swap']:<18.4f} | {res['count_swap']:<8}")
    print("="*80)
    print("✅ All tasks completed.")