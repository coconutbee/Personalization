# Personalization Evaluation Benchmark

此專案用於評估「人像個人化生成」的結果是否符合文字提示與身份一致性。流程涵蓋表情辨識、情境一致性、性別、姿態與 ID 相似度，並以加權方式輸出總分報表。

## 專案任務
1. 由提示文字自動產生 GT 標註：表情、性別、姿態。
2. 以 VLM 進行表情分類與情境一致性評分。
3. 以 MiVOLO 進行性別判斷。
4. 以姿態模型預測頭/身體姿勢，對照 GT 姿態。
5. 以 AdaFace 計算 ID 相似度 (cosine similarity)。
6. 依權重計算最終分數並輸出報表。

## 專案配置與資料夾結構
- prompts.json：輸入提示文字與影像清單。
- faceswap_results/：各方法生成/換臉後的影像輸出。
  - pixart/、janus/、infinity/、showo2/：不同方法的結果資料夾。
  - reference/：ID 相似度參考人臉影像。
- output/：備用輸出路徑 (當 method 未匹配時使用)。
- run.sh：一鍵執行整體評估流程。
- gt.json：由 prompts.json 產生的 GT 標註檔。
- gt_scored.json：計分後的完整輸出。
- final_scores.csv：彙整報表。

> 影像命名規則
> - 預設使用 0_{image}.png 作為生成結果檔名，例如 image=1.jpg → 0_1.png。
> - 腳本會嘗試多種副檔名與前綴，但建議維持 0_{id}.png 以避免找不到圖檔。

## 輸入格式

### prompts.json
```json
[
  {"id": 0, "image": "0.jpg", "prompt": "A boy looks upward..."}
]
```
欄位說明：
- id：樣本編號。
- image：對應影像檔名 (用於定位生成圖檔)。
- prompt：文字提示。

### gt.json (由 gt_maker.py 產生)
在原始欄位基礎上新增：
- gt_expression：表情標籤 (happy/surprise/confuse/neutral/sad/others)
- gt_gender：性別標籤 (Male/Female/Both/Unknown)
- gt_pose：姿態標籤 (Frontal/Head_Turn_Left/…)

## 輸出結果

### gt.json (更新欄位)
流程會逐步寫入以下欄位：
- vlm_expression：VLM 表情推論結果
- expression_correct：表情是否正確 (0/1)
- scenario_reasoning：情境一致性推論說明
- scenario_score：情境一致性分數 (0.0~1.0)
- mivolo_gender：MiVOLO 性別推論
- gender_correct：性別是否正確 (0/1)
- pose_prediction：姿態推論結果
- pose_correct：姿態是否正確 (0/1)
- id_similarity：ID 相似度 (0.0~1.0)
- final_score：最終加權分數 (0~100)

### final_scores.csv
彙整每張圖的分數與各子項結果，並計算整體平均。

## 評分權重
總分以 $0\sim100$ 計算，權重如下：
- Expression：0.17
- Scenario：0.17
- Gender：0.17
- Pose：0.17
- ID Similarity：0.32

計分實作請見 [scoring.py](scoring.py)。

## Inference 方式

### 1) 一鍵執行
```bash
bash run.sh
```
run.sh 依序執行：
1. gt_maker.py：由 prompts.json 生成 gt.json
2. exps_scenario.py：InternVL 表情與情境一致性
3. MiVOLO/gender_label.py：性別判斷
4. AdaFace/inference.py：ID 相似度
5. pose/eval_pose.py：姿態評估
6. scoring.py：加權計分並輸出報表

### 2) 指定方法推論
以下腳本支援 method 參數 (pixart/janus/infinity/showo2)，用於切換影像來源資料夾：
```bash
python exps_scenario.py --method pixart --json gt.json
python MiVOLO/gender_label.py --method pixart --json gt.json
python AdaFace/inference.py --method pixart --json gt.json
python pose/eval_pose.py --method pixart --json gt.json
```

### 3) 僅計分
```bash
python scoring.py --json gt.json --csv final_scores.csv
```

## 重要依賴與模型
- InternVL (lmdeploy pipeline)：表情分類 + 情境一致性
- MiVOLO：性別判斷
- AdaFace：ID 相似度
- Pose 模型：姿態分類

請依需求安裝依賴，必要套件版本請見 requirements.txt，部分腳本會切換 timm 版本以相容模型。