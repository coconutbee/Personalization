#!/bin/bash

# Auto face swap script
# Automatically processes images from target folder
# Uses image filenames to map to gender categories (boy/girl/man/woman)
# Each image will swap with 5 faces from corresponding gender category

cd "$(dirname "$0")"

python face_swap.py \
    --auto_mode \
    --ref_face_dir ../REF_FACE \
    --ref_image_dir ../pixart_test \
    --output_dir batch_faceswap_output_auto \
    --frame_num 21 \
    --size 512*512 \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale_img 4.0 \
    --base_seed 42 \
    --skip_existing
