conda activate pslz
python gt_maker.py 
python exps_scenario.py --json gt.json
pip install timm==0.8.13.dev0
python MiVOLO/gender_label.py --json gt.json
python AdaFace/inference.py --json gt.json
pip install timm==0.6.13
python pose/eval_pose_v2.py --json gt.json
python scoring.py --json gt.json

### CLIP, DINO, EvalMuse 
python general_scoring.py --json gt.json
python EvalMuse/eval.py --json gt.json