#!/bin/bash

# 1. 修正變數賦值（移除空格）
# JSON="pose_prompt_pixart.json"
JSON="pose_prompt_uniprotrait.json"

# t2i="/media/ee303/disk1/PuLID/pulid_pslz"
# t2i="/media/ee303/disk2/style_generation/PhotoMaker/pose_prompt_ph2"
# t2i="/media/ee303/disk1/PuLID/pose_prompt_pulid"
# t2i="/media/ee303/disk2/JACK/pixart_pose_only"
t2i="/media/ee303/disk2/JACK/UniPortrait_PoseOnly"
ref="/media/ee303/disk2/JACK/reference"

# name="pose_prompt_pixart"
name="pose_prompt_uniprotrait"

python json_maker_a.py --folder "$t2i" --output "$JSON"
python AdaFace/inference_v2_a.py --json "$JSON" --name "$name" --t2i "$t2i" --ref "$ref"
python gt_maker.py --input "$JSON" --output "$JSON"
python pose/eval_paul.py --t2i "$t2i" --name "$name" --ref "$ref" --json "$JSON"
python exps_scenario_v2_a.py --name "$name" --json "$JSON" --ref "$ref" --t2i "$t2i" --mode t2i
python general_scoring_v2_a.py --json "$JSON" --name "$name" --ref "$ref" --t2i "$t2i" --mode t2i
python scoring_v2_a.py --json "$JSON" --name "$name" --ref "$ref" --t2i "$t2i" --mode t2i
python EvalMuse/eval_v2_a.py --json "$JSON" --name "$name" --t2i "$t2i" --mode t2i