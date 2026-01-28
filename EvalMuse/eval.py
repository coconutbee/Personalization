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

# --- Patch (防止 import error) ---
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
# ---------------------

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def custom_load_model(model_path):
    # 維持 364 解析度設定
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
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    target_vocab_size = len(tokenizer)

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
    
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

def load_manual_processors():
    vis_cfg = OmegaConf.create({"name": "blip_image_eval", "image_size": 364})
    text_cfg = OmegaConf.create({"name": "blip_caption"})
    vis_processor = load_processor("blip_image_eval", cfg=vis_cfg)
    text_processor = load_processor("blip_caption", cfg=text_cfg)
    return {"eval": vis_processor}, {"eval": text_processor}

def eval(args):
    model = custom_load_model(args.model_path)
    model.eval()
    vis_processors, text_processors = load_manual_processors()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_dir = "" 
    if args.method == 'pixart': dataset_dir = "./faceswap_results/pixart"
    elif args.method == 'infinity': dataset_dir = "./faceswap_results/infinity"
    elif args.method == 'janus': dataset_dir = "./faceswap_results/janus"
    elif args.method == 'showo2': dataset_dir = "./faceswap_results/showo2"
    
    print(f">>> Dataset Dir: {dataset_dir}")

    result_list = []
    print(">>> Start Evaluation (Extracting ALL word scores)...")
    for i, item in enumerate(tqdm(data)):
        prompt = item['prompt']
        img_name = f"0_{i}.jpg"
        image_path = os.path.join(dataset_dir, img_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(dataset_dir, f"0_{i}.png")
        if not os.path.exists(image_path):
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).to(device)
            clean_prompt = text_processors["eval"](prompt)
            prompt_ids = tokenizer(clean_prompt).input_ids

            torch.cuda.empty_cache()
            with torch.no_grad():
                # alignment_score_output: 整體分數 / scores: 每個 token 的分數序列
                alignment_score_output, scores = model.element_score(image.unsqueeze(0), [clean_prompt])

            # 1. 處理整體分數 (崩潰修復)
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

            # --- [NEW] 2. 提取「每一個字」的分數 ---
            # scores 的形狀通常是 (1, seq_len) 或 (seq_len)
            scores_tensor = scores.squeeze() # 轉成 1D Tensor
            
            # 確保長度一致 (通常 scores 是跟隨 prompt_ids 的)
            min_len = min(len(prompt_ids), len(scores_tensor))
            
            word_scores = []
            for idx in range(min_len):
                token_id = prompt_ids[idx]
                token_str = tokenizer.decode([token_id]) # 將 ID 轉回文字
                token_score = scores_tensor[idx].item()
                
                # 過濾掉特殊的開始/結束符號 (可選，這裡保留以便除錯)
                if token_str in ['[CLS]', '[SEP]', '[DEC]']: continue
                
                word_scores.append([token_str, token_score])
            # ----------------------------------------

            # 3. 為了兼容性，保留原本的 result 欄位
            item['fga_alignment_score'] = f"{final_score:.2f}"
            item['all_token_scores'] = [[word, round(score, 2)] for word, score in word_scores] # 這裡存了所有字的細粒度分數
            
            result_list.append(item)
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)
    print(f">>> Results saved to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='gt.json')
    parser.add_argument('--method', type=str, default='pixart')
    parser.add_argument('--save_path', type=str, default='EM_results/fga_word_scores.json')
    parser.add_argument('--model_path', type=str, default='./EvalMuse/fga_blip2.pth')
    args = parser.parse_args()
    eval(args)