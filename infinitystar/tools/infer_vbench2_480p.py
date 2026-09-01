# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
#
# VBench-2.0 video generation worker for InfinityStar.
# Generates videos across VBench-2.0 dimensions with multi-GPU task distribution.
# Uses the EXACT same model config and inference pipeline as infer_video_480p.py.
#
# Usage (single GPU):
#     CUDA_VISIBLE_DEVICES=0 python tools/infer_vbench2_480p.py \
#         --worker_id 0 --num_workers 1 --save_dir /path/to/output
#
# Usage (multi-GPU, launched by run_vbench2_pipeline.sh):
#     CUDA_VISIBLE_DEVICES=$gpu python tools/infer_vbench2_480p.py \
#         --worker_id $i --num_workers $N --save_dir /path/to/output [options]

import sys
import os
import os.path as osp
import json
import time
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import gen_one_example, save_video
from tools.infer_video_480p import InferencePipe, build_parameters
from infinity.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from infinity.utils.arg_util import Args


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
VBENCH_BASE_DIR = osp.join(osp.dirname(SCRIPT_DIR), 'VBench', 'VBench-2.0', 'prompts')
VBENCH_PROMPT_DIR = osp.join(VBENCH_BASE_DIR, 'prompt')
VBENCH_AUG_PROMPT_DIR = osp.join(VBENCH_BASE_DIR, 'prompt_aug', 'VBench2_aug_prompt')

ALL_DIMENSIONS = [
    "Camera_Motion", "Complex_Landscape", "Complex_Plot", "Composition",
    "Diversity", "Dynamic_Spatial_Relationship", "Dynamic_Attribute",
    "Motion_Order_Understanding", "Human_Interaction", "Human_Anatomy",
    "Human_Identity", "Human_Clothes", "Instance_Preservation",
    "Material", "Mechanics", "Thermotics", "Multi-View_Consistency",
    "Motion_Rationality",
]


def build_task_list(prompt_dir, dimensions, num_videos=3, num_videos_diversity=20,
                    aug_prompt_dir=None, max_prompts_per_dim=0, prompt_seed=42):
    """Build a flat list of (dimension, original_prompt, aug_prompt, video_index) tasks.

    Diversity dimension requires 20 videos per prompt; all others require 3.
    When aug_prompt_dir is provided, augmented prompts are loaded and paired
    with original prompts (filenames always use original prompts).

    If max_prompts_per_dim > 0, randomly sample that many prompts per dimension
    using prompt_seed for reproducibility.
    """
    import random

    tasks = []
    for dim in dimensions:
        prompt_file = osp.join(prompt_dir, f'{dim}.txt')
        with open(prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        aug_prompts = None
        if aug_prompt_dir is not None:
            aug_file = osp.join(aug_prompt_dir, f'{dim}.txt')
            if osp.exists(aug_file):
                with open(aug_file, 'r') as f:
                    aug_prompts = [line.strip() for line in f if line.strip()]
                assert len(aug_prompts) == len(prompts), (
                    f'{dim}: original has {len(prompts)} prompts but aug has {len(aug_prompts)}')

        if max_prompts_per_dim > 0 and max_prompts_per_dim < len(prompts):
            original_count = len(prompts)
            rng = random.Random(prompt_seed)
            indices = sorted(rng.sample(range(len(prompts)), max_prompts_per_dim))
            prompts = [prompts[i] for i in indices]
            if aug_prompts is not None:
                aug_prompts = [aug_prompts[i] for i in indices]
            print(f'[build_task_list] {dim}: sampled {len(prompts)}/{original_count} prompts (seed={prompt_seed})')

        n = num_videos_diversity if dim == 'Diversity' else num_videos
        for p_idx, prompt in enumerate(prompts):
            aug = aug_prompts[p_idx] if aug_prompts else prompt
            for idx in range(n):
                tasks.append((dim, prompt, aug, idx))
    return tasks


def perform_inference_vbench(pipe, prompt_text, seed, duration, args, output_path):
    """Identical generation logic to infer_video_480p.perform_inference (text-to-video),
    but saves the result directly to output_path for VBench-2.0 naming convention.
    """
    num_frames = duration * 16 + 1

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn][
        'pt2scale_schedule'][(num_frames - 1) // 4 + 1]

    args.first_full_spatial_size_scale_index = \
        get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(
        scale_schedule, args.first_full_spatial_size_scale_index, args)
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn][
        'pt2scale_schedule'][(num_frames - 1) // 4 + 1]

    cond_anneal_params, scale_travel_params, _ = build_parameters(
        args, scale_schedule)
    tau = cond_anneal_params['tau_list']
    cfg_list = cond_anneal_params['cfg_list']
    noise_list = cond_anneal_params['noise_scales']

    gt_leak, gt_ls_Bl = -1, None

    negative_prompt = ""
    prompt = f'{prompt_text}, Close-up on big objects, emphasize scale and detail'
    if args.append_duration2caption:
        prompt = f'<<<t={duration}s>>>' + prompt

    for b in pipe.infinity.unregistered_blocks:
        b.attn.kv_caching(False)

    start_time = time.time()
    with torch.cuda.amp.autocast(
        enabled=True, dtype=torch.bfloat16, cache_enabled=True
    ), torch.no_grad():
        generated_image, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt=negative_prompt,
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=cfg_list,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=noise_list,
            # [DiverseVAR] noise injection params
            noise_type=cond_anneal_params['noise_type'],
            noise_scales=cond_anneal_params['noise_scales'],
            noise_stage=cond_anneal_params['noise_stage'],
            noise_rescale=cond_anneal_params['noise_rescale'],
            noise_seed=cond_anneal_params['noise_seed'],
            # [DiverseVAR] keyframe scale-travel params
            repeat_num=scale_travel_params['repeat_num'],
            encode_stage=scale_travel_params['encode_stage'],
            restart_stage=scale_travel_params['restart_stage'],
            cfg_refine=scale_travel_params['cfg_refine'],
            # [DiverseVAR] temporal scale-travel params
            repeat_num_video=scale_travel_params['repeat_num_video'],
            encode_stage_video=scale_travel_params['encode_stage_video'],
            restart_stage_video=scale_travel_params['restart_stage_video'],
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)

    elapsed_time = time.time() - start_time

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    save_video(generated_image.cpu().numpy(), fps=args.fps, save_filepath=output_path)

    return elapsed_time


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # CLI argument parsing                                                #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description='VBench-2.0 video generation with InfinityStar (multi-GPU worker)')

    # ---- worker distribution ----
    parser.add_argument('--worker_id', type=int, required=True,
                        help='This worker\'s ID (0-indexed)')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Total number of parallel workers')

    # ---- VBench-2.0 settings ----
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Root output directory for VBench-2.0 videos '
                             '(videos saved as {save_dir}/{dimension}/{prompt}-{idx}.mp4)')
    parser.add_argument('--checkpoints_dir', type=str, default='pretrained/InfinityStar',
                        help='Directory containing the downloaded InfinityStar checkpoints')
    parser.add_argument('--vbench_prompt_dir', type=str, default=VBENCH_PROMPT_DIR,
                        help='Directory containing VBench-2.0 per-dimension prompt .txt files')
    parser.add_argument('--use_aug_prompt', type=int, default=0, choices=[0, 1],
                        help='Use augmented prompts for generation (filenames still use original)')
    parser.add_argument('--vbench_aug_prompt_dir', type=str, default=VBENCH_AUG_PROMPT_DIR,
                        help='Directory containing VBench-2.0 augmented prompt .txt files')
    parser.add_argument('--dimensions', nargs='+', default=None,
                        help='Subset of dimensions to generate (default: all 18)')
    parser.add_argument('--num_videos', type=int, default=3,
                        help='Videos per prompt for non-Diversity dimensions')
    parser.add_argument('--num_videos_diversity', type=int, default=20,
                        help='Videos per prompt for Diversity dimension')
    parser.add_argument('--max_prompts_per_dim', type=int, default=0,
                        help='Max prompts per dimension (0 = use all)')
    parser.add_argument('--prompt_seed', type=int, default=42,
                        help='Seed for random prompt subsampling')

    # ---- generation ----
    parser.add_argument('--duration', type=int, default=5, choices=[5, 10],
                        help='Video duration in seconds')

    # ---- model tuning ----
    parser.add_argument('--cfg', type=float, default=34,
                        help='APG scale')
    parser.add_argument('--tau_image', type=float, default=1.0,
                        help='Sampling temperature for image (semantic) scales')
    parser.add_argument('--tau_video', type=float, default=0.4,
                        help='Sampling temperature for video (detail) scales')

    # ---- DiverseVAR: noise type + shared ----
    parser.add_argument('--noise_type', type=str, default='none',
                        choices=['none', 'cads', 'sos', 'prefix', 'random'],
                        help='Type of noise injection for diverse generation')
    parser.add_argument('--noise_rescale', type=float, default=0.0,
                        help='Rescaling factor applied after noise injection')
    parser.add_argument('--noise_seed', type=int, default=730,
                        help='Seed for the noise generator')

    # ---- DiverseVAR: keyframe (image) noise ----
    parser.add_argument('--noise_scale', type=float, default=0.0,
                        help='[image] Starting noise magnitude (alias for --noise_scale_image)')
    parser.add_argument('--noise_scale_end', type=float, default=0.0,
                        help='[image] Ending noise magnitude')
    parser.add_argument('--noise_stage', type=int, default=0,
                        help='[image] Number of keyframe scales to apply noise over')
    parser.add_argument('--anneal_func', type=str, default='linear',
                        choices=['const', 'linear', 'cosine', 'exp', 'A_shape', 'V_shape'],
                        help='[image] Annealing schedule for keyframe noise')
    parser.add_argument('--noise_scale_image', type=float, default=None,
                        help='[image] Starting noise magnitude (overrides --noise_scale)')
    parser.add_argument('--noise_scale_end_image', type=float, default=None,
                        help='[image] Ending noise magnitude (overrides --noise_scale_end)')
    parser.add_argument('--noise_stage_image', type=int, default=None,
                        help='[image] Number of keyframe scales (overrides --noise_stage)')
    parser.add_argument('--anneal_func_image', type=str, default=None,
                        choices=['const', 'linear', 'cosine', 'exp', 'A_shape', 'V_shape'],
                        help='[image] Annealing schedule (overrides --anneal_func)')

    # ---- DiverseVAR: temporal (video) noise ----
    parser.add_argument('--noise_scale_video', type=float, default=0.0,
                        help='[video] Starting noise magnitude for temporal scales')
    parser.add_argument('--noise_scale_end_video', type=float, default=0.0,
                        help='[video] Ending noise magnitude for temporal scales')
    parser.add_argument('--noise_stage_video', type=int, default=0,
                        help='[video] Number of temporal scales to apply noise over')
    parser.add_argument('--anneal_func_video', type=str, default='linear',
                        choices=['const', 'linear', 'cosine', 'exp', 'A_shape', 'V_shape'],
                        help='[video] Annealing schedule for temporal noise')

    # ---- DiverseVAR: keyframe scale-travel ----
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='[image] Scale-travel passes for keyframe (1 = disabled)')
    parser.add_argument('--encode_stage', type=int, default=0,
                        help='[image] Absolute keyframe scale index for re-encoding')
    parser.add_argument('--restart_stage', type=int, default=0,
                        help='[image] Absolute keyframe scale index to restart from')

    # ---- DiverseVAR: temporal scale-travel ----
    parser.add_argument('--repeat_num_video', type=int, default=1,
                        help='[video] Scale-travel passes for temporal part (1 = disabled)')
    parser.add_argument('--encode_stage_video', type=int, default=0,
                        help='[video] Temporal scale index (relative) for re-encoding')
    parser.add_argument('--restart_stage_video', type=int, default=0,
                        help='[video] Temporal scale index (relative) to restart from')

    cli = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Fixed model config (identical to infer_video_480p.py)               #
    # ------------------------------------------------------------------ #
    checkpoints_dir = cli.checkpoints_dir

    args = Args()
    args.pn                     = '0.40M'
    args.fps                    = 16
    args.video_frames           = cli.duration * 16 + 1
    args.model_path             = os.path.join(checkpoints_dir, 'infinitystar_8b_480p_weights')
    args.checkpoint_type        = 'torch_shard'
    args.vae_path               = os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt      = os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.videovae               = 10
    args.model_type             = 'infinity_qwen8b'
    args.text_channels          = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16                   = 1
    args.use_apg                = 1
    args.use_cfg                = 0
    args.apg_norm_threshold     = 0.05
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    args.append_duration2caption = 1
    args.use_two_stage_lfq      = 1
    args.detail_scale_min_tokens = 350
    args.semantic_scales        = 11
    args.max_repeat_times       = 10000

    # ---- override with CLI values ----
    args.cfg             = cli.cfg
    args.tau_image       = cli.tau_image
    args.tau_video       = cli.tau_video

    # build_parameters needs prompt_txt and save_dir for its internal path construction;
    # we discard the returned save_path and save to VBench-2.0 dimension folders instead.
    args.prompt_txt = 'vbench2'
    args.save_dir   = osp.join(cli.save_dir, '.params_cache')

    # DiverseVAR: noise type + shared
    args.noise_type    = cli.noise_type
    args.noise_rescale = cli.noise_rescale
    args.noise_seed    = cli.noise_seed

    # DiverseVAR: keyframe noise (legacy aliases; explicit *_image overrides follow)
    args.noise_scale     = cli.noise_scale
    args.noise_scale_end = cli.noise_scale_end
    args.noise_stage     = cli.noise_stage
    args.anneal_func     = cli.anneal_func
    if cli.noise_scale_image is not None:
        args.noise_scale_image = cli.noise_scale_image
    if cli.noise_scale_end_image is not None:
        args.noise_scale_end_image = cli.noise_scale_end_image
    if cli.noise_stage_image is not None:
        args.noise_stage_image = cli.noise_stage_image
    if cli.anneal_func_image is not None:
        args.anneal_func_image = cli.anneal_func_image

    # DiverseVAR: temporal noise
    args.noise_scale_video     = cli.noise_scale_video
    args.noise_scale_end_video = cli.noise_scale_end_video
    args.noise_stage_video     = cli.noise_stage_video
    args.anneal_func_video     = cli.anneal_func_video

    # DiverseVAR: keyframe scale-travel
    args.repeat_num    = cli.repeat_num
    args.encode_stage  = cli.encode_stage
    args.restart_stage = cli.restart_stage

    # DiverseVAR: temporal scale-travel
    args.repeat_num_video    = cli.repeat_num_video
    args.encode_stage_video  = cli.encode_stage_video
    args.restart_stage_video = cli.restart_stage_video

    # ------------------------------------------------------------------ #
    # Build VBench-2.0 task list and distribute across workers            #
    # ------------------------------------------------------------------ #
    dimensions = cli.dimensions if cli.dimensions else ALL_DIMENSIONS
    aug_dir = cli.vbench_aug_prompt_dir if cli.use_aug_prompt else None
    all_tasks = build_task_list(
        cli.vbench_prompt_dir, dimensions,
        cli.num_videos, cli.num_videos_diversity,
        aug_prompt_dir=aug_dir,
        max_prompts_per_dim=cli.max_prompts_per_dim,
        prompt_seed=cli.prompt_seed,
    )

    my_tasks = [t for i, t in enumerate(all_tasks) if i % cli.num_workers == cli.worker_id]

    print(f'[Worker {cli.worker_id}/{cli.num_workers}] '
          f'Total tasks: {len(all_tasks)}, assigned: {len(my_tasks)}')

    # ------------------------------------------------------------------ #
    # Load models once (shared across all tasks on this GPU)              #
    # ------------------------------------------------------------------ #
    pipe = InferencePipe(args)

    # ------------------------------------------------------------------ #
    # Generate videos                                                     #
    # ------------------------------------------------------------------ #
    inference_times = []
    errors = []

    for task_idx, (dim, original_prompt, aug_prompt, video_idx) in enumerate(
        tqdm(my_tasks, desc=f'Worker {cli.worker_id}')
    ):
        out_dir = osp.join(cli.save_dir, dim)
        out_path = osp.join(out_dir, f'{original_prompt[:180]}-{video_idx}.mp4')

        if osp.exists(out_path):
            continue

        prompt_for_generation = aug_prompt

        try:
            elapsed = perform_inference_vbench(
                pipe, prompt_for_generation, video_idx,
                cli.duration, args, out_path,
            )
            inference_times.append(elapsed)
        except Exception as e:
            err_msg = f'[{dim}] {original_prompt[:60]}... idx={video_idx}: {e}'
            print(f'[Worker {cli.worker_id}] ERROR: {err_msg}')
            traceback.print_exc()
            errors.append(err_msg)

    # ------------------------------------------------------------------ #
    # Save worker metadata                                                #
    # ------------------------------------------------------------------ #
    meta = {
        'worker_id': cli.worker_id,
        'num_workers': cli.num_workers,
        'total_assigned': len(my_tasks),
        'completed': len(inference_times),
        'errors': len(errors),
        'error_details': errors,
        'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0.0,
    }
    meta_dir = osp.join(cli.save_dir, '.worker_logs')
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = osp.join(meta_dir, f'worker_{cli.worker_id}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[Worker {cli.worker_id}] Done. '
          f'{len(inference_times)} generated, {len(errors)} errors. '
          f'Metadata: {meta_path}')
