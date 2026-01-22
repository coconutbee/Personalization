# Personalization for Evaluation

## Input Data
- `source_images/`: Directory containing source images. ()
- `target_images/`: Directory containing target images for pose/contour reference.
- `faceswap_results/`: Directory where the swapped face images will be saved.
- `prompts.json/`: JSON file containing text prompts for image generation.

    gt_maker.py # put json file than extract gt labels(expression, gender, pose)
    ! Face_swap.py # from './pixart_outputs' get source face. and from './target_images' get target face to do face swap. save in './faceswap_results'. file name: {target_index}_{source_index}.png

    expression_label.py # use InternVL to label expression from generated images
    pip install timm==0.8.13.dev0
    MiVOLO/gender_label.py # use MiVOLO to label gender from generated images
    pip install timm==0.6.13
    pose/eval_pose.py
    AdaFace/inference.py # use Adaface to extract face embeddings and calculate similarity scores


TO DO:
pip install timm==0.6.13
X 加入POSE評分
    pose/eval_pose.py # use OpenPose to extract keypoints and calculate pose similarity scores
    
X 整理虛擬環境

X 修改Scenario Check程式碼，讓internvl專注在scenario check(與exps整合)
    expsAscn.py # integrate expression and scenario check, calculate scenario scores
X 加入Adaface評分
    adaface_label.py # use Adaface to extract face embeddings and calculate similarity scores
建立最終分數報告程式碼
    final_report.py # 為每一項分數加入權重並計算總分
整合所有評分程式碼
    run.sh # bash script to run all evaluation scripts and generate final report