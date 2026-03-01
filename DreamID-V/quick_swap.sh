#!/bin/bash

# Quick face swap script for two uploaded images
# Usage: bash quick_swap.sh <ref_image> <target_image>

if [ "$#" -ne 2 ]; then
    echo "Usage: bash quick_swap.sh <reference_face_image> <target_image>"
    echo "Example: bash quick_swap.sh man_face.jpg woman_portrait.jpg"
    exit 1
fi

REF_IMAGE="$1"
TARGET_IMAGE="$2"

# Check if files exist
if [ ! -f "$REF_IMAGE" ]; then
    echo "Error: Reference image not found: $REF_IMAGE"
    exit 1
fi

if [ ! -f "$TARGET_IMAGE" ]; then
    echo "Error: Target image not found: $TARGET_IMAGE"
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
cp "$TARGET_IMAGE" "$TEMP_DIR/"

echo "Reference face: $REF_IMAGE"
echo "Target image: $TARGET_IMAGE"
echo "Processing..."

# Activate conda environment and run face swap
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dreamid

cd "$(dirname "$0")"

python face_swap.py \
    --ref_image "$REF_IMAGE" \
    --target_folder "$TEMP_DIR" \
    --output_dir quick_swap_output \
    --frame_num 21 \
    --size 512*512 \
    --sample_steps 50

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "Done! Check output in: quick_swap_output/"
