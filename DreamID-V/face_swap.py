#!/usr/bin/env python3
"""
Batch face swapping script with single reference face
Process multiple target images with one reference face image
Based on generate_dreamidv_image.py with batch processing optimization
"""

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import glob
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import cv2
import numpy as np
from PIL import Image

import dreamidv_wan_faster
from dreamidv_wan_faster.configs import WAN_CONFIGS, SIZE_CONFIGS
from dreamidv_wan_faster.utils.utils import str2bool
from torchvision.utils import save_image

# Import face mask generation utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'express_adaption'))
from express_adaption.media_pipe import FaceMeshDetector


def generate_face_mask_for_image(image_path, output_path):
    """Generate face mask for a single image using MediaPipe"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    h, w = img.shape[:2]
    detector = FaceMeshDetector()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    motion_image, face_result = detector(img_rgb)
    
    if face_result is None:
        print(f"Warning: No face detected in {image_path}, creating empty mask")
        mask = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        lmks = face_result['lmks'].astype(np.float32)
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        points = []
        for i in range(lmks.shape[0]):
            x = int(lmks[i, 0] * w)
            y = int(lmks[i, 1] * h)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    cv2.imwrite(output_path, mask)
    return output_path


def convert_image_to_video(image_path, output_video_path, fps=24, num_frames=1):
    """Convert a single image to a multi-frame video by repeating the frame"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    for _ in range(num_frames):
        out.write(img)
    
    out.release()
    return output_video_path


def get_target_images(target_folder, recursive=False):
    """Get all image files from target folder"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    if recursive:
        for ext in image_extensions:
            pattern = os.path.join(target_folder, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in image_extensions:
            pattern = os.path.join(target_folder, ext)
            image_files.extend(glob.glob(pattern))
    
    # Filter out temp files
    image_files = [f for f in image_files if 'temp_generated' not in f]
    return sorted(image_files)


def map_prompt_to_gender(prompt):
    """Map prompt to gender category (boy/girl/man/woman)"""
    prompt_lower = prompt.lower()
    
    # Check for specific keywords in order of priority
    if 'boy' in prompt_lower:
        return 'boy'
    elif 'girl' in prompt_lower:
        return 'girl'
    elif 'lady' in prompt_lower:
        return 'woman'
    elif 'woman' in prompt_lower:
        return 'woman'
    elif 'man' in prompt_lower:
        return 'man'
    else:
        # Default to 'man' if no match found
        return 'man'


def load_prompts(prompt_file):
    """Load prompts from file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Batch face swapping with automatic prompt-based gender mapping"
    )
    # Create mutually exclusive group for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--auto_mode",
        action="store_true",
        help="Automatic mode: read images from target folder, use filenames to map genders"
    )
    mode_group.add_argument(
        "--ref_image",
        type=str,
        help="Single reference face image (face to swap from)"
    )
    mode_group.add_argument(
        "--ref_folder",
        type=str,
        help="Folder containing multiple reference face images"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Path to prompt file (deprecated, not used in auto_mode)"
    )
    parser.add_argument(
        "--ref_face_dir",
        type=str,
        default="../REF_FACE",
        help="Path to REF_FACE directory containing boy/girl/man/woman folders"
    )
    parser.add_argument(
        "--ref_image_dir",
        type=str,
        help="Path to directory containing target images (used in auto_mode)"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        help="Folder containing target images to process (not used in auto_mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_faceswap_output",
        help="Directory to save face-swapped results"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process target folder recursively (include subdirectories)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="512*512",
        help="Output size (e.g., 1024*1024, 512*512)"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=21,
        help="Number of frames for generation (higher = better quality, default: 13)"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Sampling steps (default: 50)"
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5.0,
        help="Sampling shift factor (default: 5.0)"
    )
    parser.add_argument(
        "--sample_guide_scale_img",
        type=float,
        default=4.0,
        help="Classifier free guidance scale (default: 4.0)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="models/Wan2.1-T2V-1.3B",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--dreamidv_ckpt",
        type=str,
        default="models/dreamidv_faster.pth",
        help="DreamID-V checkpoint path"
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="Offload model to CPU after each forward pass"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip already processed images"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    
    # Handle different modes
    if args.auto_mode:
        # Automatic mode: read images directly from target folder
        if not os.path.exists(args.ref_face_dir):
            print(f"Error: REF_FACE directory not found: {args.ref_face_dir}")
            return
        if not args.ref_image_dir or not os.path.exists(args.ref_image_dir):
            print(f"Error: ref_image_dir not found: {args.ref_image_dir}")
            print("Usage: --ref_image_dir /path/to/target/images")
            return
        
        # Get all target images from the target folder
        target_images = get_target_images(args.ref_image_dir, recursive=False)
        if not target_images:
            print(f"Error: No images found in {args.ref_image_dir}")
            return
        
        # Build task list: each task = (image_path, filename, gender, ref_faces)
        tasks = []
        for target_image in target_images:
            # Extract filename (without extension) to use as prompt
            filename = os.path.basename(target_image)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Map filename to gender
            gender = map_prompt_to_gender(name_without_ext)
            
            # Get 5 reference faces from REF_FACE/{gender}/
            ref_face_folder = os.path.join(args.ref_face_dir, gender)
            if not os.path.exists(ref_face_folder):
                print(f"Warning: {ref_face_folder} not found, skipping {filename}")
                continue
            
            ref_faces = get_target_images(ref_face_folder, recursive=False)
            if len(ref_faces) < 5:
                print(f"Warning: {ref_face_folder} has only {len(ref_faces)} faces, expected 5")
            ref_faces = ref_faces[:5]  # Take first 5
            
            if not ref_faces:
                print(f"Warning: No reference faces found for gender '{gender}', skipping {filename}")
                continue
            
            tasks.append({
                'target_image': target_image,
                'filename': name_without_ext,
                'gender': gender,
                'ref_faces': ref_faces
            })
        
        if not tasks:
            print("Error: No valid tasks to process")
            return
        
        # Calculate total combinations
        total_combinations = sum(len(task['ref_faces']) for task in tasks)
        
        print(f"\n{'='*60}")
        print(f"AUTO MODE: BATCH FACE SWAP CONFIGURATION")
        print(f"{'='*60}")
        print(f"Target folder: {args.ref_image_dir}")
        print(f"Found {len(target_images)} target images")
        print(f"Processing {len(tasks)} valid images")
        print(f"Total combinations: {total_combinations} (each image × 5 faces)")
        print(f"Output directory: {args.output_dir}")
        print(f"\nGender distribution:")
        gender_count = {}
        for task in tasks:
            gender_count[task['gender']] = gender_count.get(task['gender'], 0) + 1
        for gender, count in sorted(gender_count.items()):
            print(f"  {gender}: {count} images")
        print(f"{'='*60}\n")
        
    else:
        # Manual mode: original behavior
        ref_images = []
        if args.ref_image:
            # Single reference image mode
            if not os.path.exists(args.ref_image):
                print(f"Error: Reference image not found: {args.ref_image}")
                return
            ref_images = [args.ref_image]
        else:
            # Multiple reference images mode
            if not os.path.exists(args.ref_folder):
                print(f"Error: Reference folder not found: {args.ref_folder}")
                return
            ref_images = get_target_images(args.ref_folder, recursive=False)
            if not ref_images:
                print(f"Error: No images found in {args.ref_folder}")
                return
        
        if not args.target_folder or not os.path.exists(args.target_folder):
            print(f"Error: Target folder not found: {args.target_folder}")
            return
        
        # Get all target images
        target_images = get_target_images(args.target_folder, args.recursive)
        
        if not target_images:
            print(f"Error: No images found in {args.target_folder}")
            return
        
        print(f"\n{'='*60}")
        print(f"MANUAL MODE: BATCH FACE SWAP CONFIGURATION")
        print(f"{'='*60}")
        if args.ref_image:
            print(f"Reference face: {args.ref_image}")
        else:
            print(f"Reference folder: {args.ref_folder}")
            print(f"Found {len(ref_images)} reference faces")
        print(f"Target folder: {args.target_folder}")
        print(f"Found {len(target_images)} target images")
        print(f"Total combinations: {len(ref_images)} × {len(target_images)} = {len(ref_images) * len(target_images)}")
        print(f"Output directory: {args.output_dir}")
        print(f"Frame number: {args.frame_num}")
        print(f"Output size: {args.size}")
        print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse size configuration
    if args.size in SIZE_CONFIGS:
        size_config = SIZE_CONFIGS[args.size]
    else:
        try:
            if '*' in args.size:
                height, width = map(int, args.size.split('*'))
            elif 'x' in args.size:
                height, width = map(int, args.size.split('x'))
            else:
                height = width = int(args.size)
            size_config = (height, width)
            print(f"Using custom size: {height}x{width}")
        except Exception as e:
            print(f"Error parsing size '{args.size}': {e}")
            print(f"Available predefined sizes: {list(SIZE_CONFIGS.keys())}")
            return
    
    # Initialize model ONCE
    print(f"\n{'='*60}")
    print("LOADING MODEL (only once for all images)")
    print(f"{'='*60}")
    
    device = 0
    cfg = WAN_CONFIGS['swapface']
    cfg.sample_fps = 24
    
    logging.info("Creating DreamID-V pipeline...")
    wan_swapface = dreamidv_wan_faster.DreamIDV(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        dreamidv_ckpt=args.dreamidv_ckpt,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    
    print("Model loaded successfully!")
    print(f"{'='*60}\n")
    
    # Process each combination of reference and target images
    success_count = 0
    skipped_count = 0
    failed_images = []
    
    # Create progress bar for all combinations
    if args.auto_mode:
        # Auto mode: process tasks (each image with 5 reference faces)
        total_combinations = sum(len(task['ref_faces']) for task in tasks)
        
        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for task in tasks:
                target_image = task['target_image']
                filename = task['filename']
                gender = task['gender']
                
                # Get target image info
                target_name = os.path.basename(target_image)
                target_basename = os.path.splitext(target_name)[0]
                target_ext = os.path.splitext(target_name)[1]
                
                for ref_face in task['ref_faces']:
                    ref_basename = os.path.splitext(os.path.basename(ref_face))[0]
                    
                    try:
                        # Generate output filename: {ref_name}_{target_name}.{ext}
                        output_filename = f"{ref_basename}_{target_basename}{target_ext}"
                        output_file = os.path.join(args.output_dir, output_filename)
                        
                        # Skip if already processed
                        if args.skip_existing and os.path.exists(output_file):
                            skipped_count += 1
                            pbar.update(1)
                            continue
                        
                        # Prepare temp directory
                        temp_dir = os.path.join(os.path.dirname(os.path.abspath(target_image)), 'temp_generated')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        target_basename_temp = os.path.splitext(os.path.basename(target_image))[0]
                        mask_path = os.path.join(temp_dir, f"{target_basename_temp}_mask.png")
                        target_video_path = os.path.join(temp_dir, f"{target_basename_temp}_video_{args.frame_num}f.mp4")
                        mask_video_path = os.path.join(temp_dir, f"{target_basename_temp}_mask_{args.frame_num}f.mp4")
                        
                        # Generate mask if not exists
                        if not os.path.exists(mask_path):
                            generate_face_mask_for_image(target_image, mask_path)
                        
                        # Convert to video format
                        if not os.path.exists(target_video_path):
                            convert_image_to_video(target_image, target_video_path, num_frames=args.frame_num)
                        
                        if not os.path.exists(mask_video_path):
                            convert_image_to_video(mask_path, mask_video_path, num_frames=args.frame_num)
                        
                        # Prepare paths: target video, mask video, reference face
                        ref_paths = [target_video_path, mask_video_path, ref_face]
                        
                        # Generate face swap
                        result_image = wan_swapface.generate(
                            'change face',
                            ref_paths,
                            size=size_config,
                            frame_num=args.frame_num,
                            shift=args.sample_shift,
                            sample_solver='unipc',
                            sampling_steps=args.sample_steps,
                            guide_scale_img=args.sample_guide_scale_img,
                            seed=args.base_seed,
                            offload_model=args.offload_model
                        )
                        
                        # Extract middle frame for best quality
                        actual_frames = result_image.shape[1]
                        middle_frame_idx = actual_frames // 2
                        single_frame = result_image[:, middle_frame_idx, :, :]
                        
                        # Save result
                        save_image(
                            single_frame,
                            output_file,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                        
                        success_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        error_msg = str(e)
                        tqdm.write(f"\nError processing {target_basename} + {ref_basename}: {error_msg}")
                        failed_images.append((f"{target_basename}_{ref_basename}", error_msg))
                        pbar.update(1)
                        
                        # Stop on CUDA errors
                        if "CUDA error" in error_msg or "CUDA" in error_msg:
                            print(f"\n{'='*60}")
                            print("CRITICAL: CUDA error detected!")
                            print("GPU memory may be corrupted. Stopping batch process.")
                            print(f"{'='*60}")
                            pbar.close()
                            raise RuntimeError("CUDA error - stopping all processing")
                        
                        continue
    else:
        # Manual mode: original behavior
        total_combinations = len(ref_images) * len(target_images)
        
        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for ref_image in ref_images:
                # Get reference image basename (without extension)
                ref_basename = os.path.splitext(os.path.basename(ref_image))[0]
                
                for target_image in target_images:
                    try:
                        # Get target image name and extension
                        target_name = os.path.basename(target_image)
                        target_basename = os.path.splitext(target_name)[0]
                        target_ext = os.path.splitext(target_name)[1]
                        
                        # Generate output filename: {ref_name}_{target_name}.{ext}
                        output_filename = f"{ref_basename}_{target_basename}{target_ext}"
                        
                        if args.recursive:
                            # Preserve relative path structure
                            rel_dir = os.path.dirname(os.path.relpath(target_image, args.target_folder))
                            output_file = os.path.join(args.output_dir, rel_dir, output_filename)
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        else:
                            output_file = os.path.join(args.output_dir, output_filename)
                        
                        # Skip if already processed
                        if args.skip_existing and os.path.exists(output_file):
                            skipped_count += 1
                            pbar.update(1)
                            continue
                        
                        # Prepare temp directory
                        temp_dir = os.path.join(os.path.dirname(os.path.abspath(target_image)), 'temp_generated')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        target_basename_temp = os.path.splitext(os.path.basename(target_image))[0]
                        mask_path = os.path.join(temp_dir, f"{target_basename_temp}_mask.png")
                        target_video_path = os.path.join(temp_dir, f"{target_basename_temp}_video_{args.frame_num}f.mp4")
                        mask_video_path = os.path.join(temp_dir, f"{target_basename_temp}_mask_{args.frame_num}f.mp4")
                        
                        # Generate mask if not exists
                        if not os.path.exists(mask_path):
                            generate_face_mask_for_image(target_image, mask_path)
                        
                        # Convert to video format (regenerate if frame_num changed)
                        if not os.path.exists(target_video_path):
                            convert_image_to_video(target_image, target_video_path, num_frames=args.frame_num)
                        
                        if not os.path.exists(mask_video_path):
                            convert_image_to_video(mask_path, mask_video_path, num_frames=args.frame_num)
                        
                        # Prepare paths: target video, mask video, reference face
                        ref_paths = [target_video_path, mask_video_path, ref_image]
                        
                        # Generate face swap
                        result_image = wan_swapface.generate(
                            'change face',
                            ref_paths,
                            size=size_config,
                            frame_num=args.frame_num,
                            shift=args.sample_shift,
                            sample_solver='unipc',
                            sampling_steps=args.sample_steps,
                            guide_scale_img=args.sample_guide_scale_img,
                            seed=args.base_seed,
                            offload_model=args.offload_model
                        )
                        
                        # Extract middle frame for best quality
                        actual_frames = result_image.shape[1]
                        middle_frame_idx = actual_frames // 2
                        single_frame = result_image[:, middle_frame_idx, :, :]
                        
                        # Save result
                        save_image(
                            single_frame,
                            output_file,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                        
                        success_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        error_msg = str(e)
                        tqdm.write(f"\nError processing {ref_image} + {target_image}: {error_msg}")
                        failed_images.append((f"{ref_basename}_{target_basename}", error_msg))
                        pbar.update(1)
                        
                        # Stop on CUDA errors
                        if "CUDA error" in error_msg or "CUDA" in error_msg:
                            print(f"\n{'='*60}")
                            print("CRITICAL: CUDA error detected!")
                            print("GPU memory may be corrupted. Stopping batch process.")
                            print(f"{'='*60}")
                            pbar.close()
                            # Use a flag to break outer loop
                            raise RuntimeError("CUDA error - stopping all processing")
                        
                        continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    if args.auto_mode:
        print(f"Mode: Automatic (filename-based)")
        print(f"Processed images: {len(tasks)}")
    else:
        print(f"Mode: Manual")
        print(f"Reference faces: {len(ref_images)}")
        print(f"Target images: {len(target_images)}")
    print(f"Successful: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(failed_images)}")
    print(f"Output directory: {args.output_dir}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img, error in failed_images:
            print(f"  - {img}")
            print(f"    Error: {error}")
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
