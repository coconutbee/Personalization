python json_maker.py --folder '/media/ee303/disk2/style_generation/diffusers/pixart_test' --output 'metadata.json'
python AdaFace/inference_v2.py --json metadata.json --name pixart --t2i /media/ee303/disk2/style_generation/diffusers/pixart_test  --swap /media/ee303/disk2/JACK/FACE_SWAPED_pixart_test --ref /media/ee303/disk2/JACK/reference
python gt_maker.py --input metadata.json --output metadata.json
python pose/eval_pose_v2.py --t2i /media/ee303/disk2/style_generation/diffusers/pixart_test --name pixart --swap /media/ee303/disk2/JACK/FACE_SWAPED_pixart_test --ref /media/ee303/disk2/JACK/reference --json metadata.json
python exps_scenario_v2.py
python general_scoring_v2.py
python scoring_v2.py 
python EvalMuse/eval_v2.py