# single-gpu
python generate_dreamidv_faster.py \
    --size 832*480 \
    --ckpt_dir models/Wan2.1-T2V-1.3B \
    --dreamidv_ckpt models/dreamidv_faster.pth  \
    --sample_steps 50 \
    --base_seed 42



# multi-gpu
torchrun --nproc_per_node=2 generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv path  \
    --sample_steps 50 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 1 \
    --base_seed 42
