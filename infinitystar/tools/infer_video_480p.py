# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import sys
import time
import numpy as np
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.video_decoder import EncodedVideoDecord
from infinity.utils.arg_util import Args
from infinity.utils.diverse_var_utils import (
    cosine_annealing, a_shape_annealing, v_shape_annealing,
    exponential_annealing, linear_annealing,
)


def _build_part_noise_scales(noise_scale, noise_scale_end, noise_stage, anneal_func, n_part):
    """Build a per-scale noise schedule for ONE generation part (image or video).

    Returns a list of length n_part whose first min(noise_stage, n_part) elements
    follow the requested annealing curve; the rest are 0.
    """
    if noise_stage <= 0 or n_part <= 0:
        return [0.0] * n_part
    stage = min(noise_stage, n_part)
    if stage == 1:
        sched = [noise_scale]
    elif anneal_func == 'const':
        sched = [noise_scale] * stage
    elif anneal_func == 'linear':
        sched = list(linear_annealing(noise_scale, noise_scale_end, stage))
    elif anneal_func == 'cosine':
        sched = list(cosine_annealing(noise_scale, noise_scale_end, stage))
    elif anneal_func == 'exp':
        sched = list(exponential_annealing(noise_scale, noise_scale_end, stage))
    elif anneal_func == 'A_shape':
        sched = list(a_shape_annealing(noise_scale, noise_scale_end, stage))
    elif anneal_func == 'V_shape':
        sched = list(v_shape_annealing(noise_scale, noise_scale_end, stage))
    else:
        sched = list(linear_annealing(noise_scale, noise_scale_end, stage))
    return sched + [0.0] * (n_part - stage)


def build_parameters(args, scale_schedule, verbose=False):
    """
    Build DiverseVAR inference parameters from args, with SEPARATE control
    for the keyframe (image, pt=1) and temporal (video, pt>1) parts.

    scale_schedule layout (5-second, 28 scales):
      scales  0..tower_split-1  belong to the keyframe part (pt=1),  n_image scales
      scales  tower_split..n-1  belong to the temporal part (pt=20), n_video scales

    Noise injection
    ---------------
      Keyframe  : noise_scale_image / noise_stage_image / anneal_func_image
                  (fall back to legacy noise_scale / noise_stage / anneal_func)
      Temporal  : noise_scale_video / noise_stage_video / anneal_func_video
                  (default: no noise)
      Both parts are built independently then concatenated into one flat
      noise_scales list [img_ns_0, ..., img_ns_N-1, vid_ns_0, ..., vid_ns_M-1]
      so the model receives a single unified schedule with the correct split.

    Scale-Travel
    ------------
      Keyframe  : repeat_num / encode_stage / restart_stage
                  (encode/restart indices are absolute, within 0..tower_split-1)
      Temporal  : repeat_num_video / encode_stage_video / restart_stage_video
                  (encode/restart indices are RELATIVE to video part start,
                   i.e. 0 = first video scale = absolute index tower_split)

    Returns:
        cond_anneal_params : dict (noise_type, noise_scales, noise_stage,
                                   noise_rescale, noise_seed, cfg_list, tau_list)
        scale_travel_params: dict (repeat_num, encode_stage, restart_stage,
                                   cfg_refine,
                                   repeat_num_video, encode_stage_video,
                                   restart_stage_video)
        save_path          : str, structured output directory
    """
    n = len(scale_schedule)
    tower_split = getattr(args, 'tower_split_index', n)
    n_image = tower_split        # keyframe scales (pt=1)
    n_video = n - tower_split    # temporal  scales (pt>1)

    # ------------------------------------------------------------------ #
    # 1. noise_scales = image_part ++ video_part                         #
    # ------------------------------------------------------------------ #
    noise_type = getattr(args, 'noise_type', 'none')

    # Keyframe noise: explicit args override the legacy aliases
    noise_scale_image     = getattr(args, 'noise_scale_image',     getattr(args, 'noise_scale',     0.0))
    noise_scale_end_image = getattr(args, 'noise_scale_end_image', getattr(args, 'noise_scale_end', 0.0))
    noise_stage_image     = getattr(args, 'noise_stage_image',     getattr(args, 'noise_stage',     0))
    anneal_func_image     = getattr(args, 'anneal_func_image',     getattr(args, 'anneal_func',     'linear'))

    # Temporal noise (default: no noise)
    noise_scale_video     = getattr(args, 'noise_scale_video',     0.0)
    noise_scale_end_video = getattr(args, 'noise_scale_end_video', 0.0)
    noise_stage_video     = getattr(args, 'noise_stage_video',     0)
    anneal_func_video     = getattr(args, 'anneal_func_video',     'linear')

    if noise_type in ('cads', 'sos', 'prefix', 'random'):
        noise_scales_image = _build_part_noise_scales(
            noise_scale_image, noise_scale_end_image, noise_stage_image, anneal_func_image, n_image)
        noise_scales_video = _build_part_noise_scales(
            noise_scale_video, noise_scale_end_video, noise_stage_video, anneal_func_video, n_video)
    else:
        noise_scale_image = noise_scale_end_image = 0.0
        noise_stage_image = 0
        anneal_func_image = 'linear'
        noise_scale_video = noise_scale_end_video = 0.0
        noise_stage_video = 0
        anneal_func_video = 'linear'
        noise_scales_image = [0.0] * n_image
        noise_scales_video = [0.0] * n_video

    # single flat list: first n_image entries = keyframe, next n_video = temporal
    noise_scales = noise_scales_image + noise_scales_video

    if verbose:
        print(f'[DiverseVAR] n_total={n}, tower_split={tower_split}, n_image={n_image}, n_video={n_video}')
        print(f'[DiverseVAR] noise_type={noise_type}')
        print(f'[DiverseVAR] image noise: scale={noise_scale_image}, scale_end={noise_scale_end_image}, stage={noise_stage_image}, anneal={anneal_func_image}')
        print(f'[DiverseVAR] video noise: scale={noise_scale_video}, scale_end={noise_scale_end_video}, stage={noise_stage_video}, anneal={anneal_func_video}')
        print(f'[DiverseVAR] noise_scales (img part) = {noise_scales_image}')
        print(f'[DiverseVAR] noise_scales (vid part) = {noise_scales_video}')
        print(f'[DiverseVAR] full noise_scales = {noise_scales}')

    # ------------------------------------------------------------------ #
    # 2. cfg_list: per-scale guidance                                    #
    # ------------------------------------------------------------------ #
    cfg = getattr(args, 'cfg', 34)
    cfg_list = [cfg] * n

    # ------------------------------------------------------------------ #
    # 3. tau_list: per-scale temperature (image / video split)           #
    # ------------------------------------------------------------------ #
    tau_image = getattr(args, 'tau_image', 1.0)
    tau_video = getattr(args, 'tau_video', 0.4)
    tau_list = [tau_image] * tower_split + [tau_video] * n_video

    if verbose:
        print(f'[DiverseVAR] cfg_list[:4]={cfg_list[:4]}, tau_list[:4]={tau_list[:4]}')

    cond_anneal_params = {
        'noise_type':    noise_type,
        'noise_scales':  noise_scales,      # flat list len=n, keyframe+temporal
        'noise_stage':   noise_stage_image, # kept for compat; model uses noise_scales directly
        'noise_rescale': getattr(args, 'noise_rescale', 0.0),
        'noise_seed':    getattr(args, 'noise_seed', 730),
        'cfg_list':      cfg_list,
        'tau_list':      tau_list,
    }

    # ------------------------------------------------------------------ #
    # 4. scale_travel_params: keyframe + temporal independently          #
    # ------------------------------------------------------------------ #
    repeat_num    = getattr(args, 'repeat_num',    1)
    encode_stage  = getattr(args, 'encode_stage',  0)
    restart_stage = getattr(args, 'restart_stage', 0)
    cfg_refine_val = getattr(args, 'cfg_refine', None)
    cfg_refine = [cfg_refine_val if cfg_refine_val is not None else cfg] * n

    # Temporal scale-travel: indices are RELATIVE to video part start
    repeat_num_video    = getattr(args, 'repeat_num_video',    1)
    encode_stage_video  = getattr(args, 'encode_stage_video',  0)
    restart_stage_video = getattr(args, 'restart_stage_video', 0)

    if verbose:
        print(f'[DiverseVAR] image ST: repeat={repeat_num}, encode={encode_stage}, restart={restart_stage}')
        print(f'[DiverseVAR] video ST: repeat={repeat_num_video}, encode={encode_stage_video}, restart={restart_stage_video} (relative)')

    scale_travel_params = {
        'repeat_num':          repeat_num,
        'encode_stage':        encode_stage,
        'restart_stage':       restart_stage,
        'cfg_refine':          cfg_refine,
        # temporal scale-travel (passed separately so the model knows which part to travel)
        'repeat_num_video':    repeat_num_video,
        'encode_stage_video':  encode_stage_video,
        'restart_stage_video': restart_stage_video,
    }

    # ------------------------------------------------------------------ #
    # 5. Structured save path                                             #
    # ------------------------------------------------------------------ #
    save_dir = getattr(args, 'save_dir', 'output')
    save_path = save_dir
    prompt_txt = getattr(args, 'prompt_txt', None)
    prompt_group = osp.splitext(osp.basename(prompt_txt))[0] if prompt_txt else 'single_prompt'
    save_path += f"/{prompt_group}"
    save_path += f"/cfg={cfg}_tau_img={tau_image}_tau_vid={tau_video}"
    save_path += f"/{noise_type}"
    # keyframe noise tag
    save_path += f"/img_{noise_scale_image}_{noise_stage_image}_{anneal_func_image}"
    # append video noise tag only when non-zero
    if noise_stage_video > 0:
        save_path += f"_vid_{noise_scale_video}_{noise_stage_video}_{anneal_func_video}"
    # scale-travel tag
    st_parts = []
    if repeat_num > 1:
        st_parts.append(f"img_st_{repeat_num}_{encode_stage}_{restart_stage}")
    if repeat_num_video > 1:
        st_parts.append(f"vid_st_{repeat_num_video}_{encode_stage_video}_{restart_stage_video}")
    save_path += "/" + ("_".join(st_parts) if st_parts else "no_scale_travel")
    os.makedirs(save_path, exist_ok=True)

    if verbose:
        print(f'[DiverseVAR] save_path: {save_path}')

    return cond_anneal_params, scale_travel_params, save_path


class InferencePipe:
    def __init__(self, args):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to('cuda')
        # load infinity
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)


def perform_inference(pipe, data, args, prompt_tag):
    
    original_prompt = data["prompt"]
    seed            = data["seed"]
    mapped_duration = data['duration']
    num_frames      = mapped_duration * 16 + 1

    image_path = data.get("image_path", None)
    video_path = data.get("video_path", None)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]

    # print(f"[DEBUG] scale_schedule: {scale_schedule}")

    # Build DiverseVAR parameters and structured save path
    cond_anneal_params, scale_travel_params, save_path = build_parameters(args, scale_schedule)
    tau       = cond_anneal_params['tau_list']
    cfg_list  = cond_anneal_params['cfg_list']
    noise_list = cond_anneal_params['noise_scales']
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    if video_path is not None:
        # Video continuation
        video = EncodedVideoDecord(video_path, os.path.basename(video_path), num_threads=0)
        duration = video._duration
        if mapped_duration != 10:
            raise ValueError('Video continuation only supports 10-second generation.')
        if duration < 5:
            raise ValueError('Input video duration must be at least 5 seconds.')
        condition_scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(81-1)//4+1]
        cond_tgt_h, cond_tgt_w = condition_scale_schedule[-1][1] * 16, condition_scale_schedule[-1][2] * 16
        raw_video, _ = video.get_clip(0, 5, 81)
        video_T3HW = [transform(Image.fromarray(frame).convert("RGB"), cond_tgt_h, cond_tgt_w) for frame in raw_video]
        video_T3HW = torch.stack(video_T3HW, 0)
        video_bcthw = video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
            pipe.vae, video_bcthw.cuda(), vae_features=None,
            self_correction=pipe.self_correction, args=args, infer_mode=True,
            dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak = 28
    elif image_path is not None:
        # Image-to-Video
        ref_image = [cv2.imread(image_path)[:, :, ::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0)
        ref_img_bcthw = ref_img_T3HW.permute(1, 0, 2, 3).unsqueeze(0)
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
            pipe.vae, ref_img_bcthw.cuda(), vae_features=None,
            self_correction=pipe.self_correction, args=args, infer_mode=True,
            dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak = 14

    negative_prompt = ""
    prompt = f'{original_prompt}, Close-up on big objects, emphasize scale and detail'
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt

    # ------------------------------------------------------------------ #
    # KV cache explicit cleanup before each inference call                #
    # Mirrors the kv_caching(True/False) lifecycle in ar_infer_infinity_  #
    # elegant: ensure no stale state leaks across seeds / prompts.        #
    # ------------------------------------------------------------------ #
    for b in pipe.infinity.unregistered_blocks:
        b.attn.kv_caching(False)

    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
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
            # [DiverseVAR] temporal scale-travel params (indices relative to video part)
            repeat_num_video=scale_travel_params['repeat_num_video'],
            encode_stage_video=scale_travel_params['encode_stage_video'],
            restart_stage_video=scale_travel_params['restart_stage_video'],
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)

    elapsed_time = time.time() - start_time

    # ------------------------------------------------------------------ #
    # Save video to structured path (mirrors Infinity/inference.py)       #
    # Layout: {save_path}/{prompt_tag}.mp4                #
    # ------------------------------------------------------------------ #
    # prompt_tag = original_prompt[:50].replace('/', '_').replace(' ', '_')
    video_save_dir = osp.join(save_path)
    os.makedirs(video_save_dir, exist_ok=True)
    gen_video_path = osp.join(video_save_dir, f'{prompt_tag}.mp4')
    save_video(generated_image.cpu().numpy(), fps=args.fps, save_filepath=gen_video_path)

    return {
        "output":       generated_image.cpu().numpy(),
        "elapsed_time": elapsed_time,
        "save_path":    gen_video_path,
        "save_dir":     save_path,
    }


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # CLI argument parsing                                                #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description='InfinityStar 480p video inference')

    # ---- generation ----
    parser.add_argument('--prompt_txt', type=str, required=True,
                        help='Path to a text file with one prompt per line '
                             '(e.g. ../prompts/example-video-prompt.txt)')
    parser.add_argument('--duration', type=int, default=5, choices=[5, 10],
                        help='Video duration in seconds')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='First seed value (inclusive)')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of seeds to generate per prompt')
    parser.add_argument('--save_dir', type=str, default='output',
                        help='Root directory for generated videos')
    parser.add_argument('--checkpoints_dir', type=str, default='pretrained/InfinityStar',
                        help='Directory containing the downloaded InfinityStar checkpoints')

    # ---- multi-GPU sharding (optional) ----
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker id for multi-GPU sharding of prompt/seed pairs')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers for sharding prompt/seed pairs')

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
    # Legacy aliases (noise_scale / noise_stage / anneal_func) apply to the keyframe part.
    # The explicit *_image variants override them when provided.
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
                        help='[image] Number of keyframe scales to apply noise over (overrides --noise_stage)')
    parser.add_argument('--anneal_func_image', type=str, default=None,
                        choices=['const', 'linear', 'cosine', 'exp', 'A_shape', 'V_shape'],
                        help='[image] Annealing schedule (overrides --anneal_func)')

    # ---- DiverseVAR: temporal (video) noise ----
    parser.add_argument('--noise_scale_video', type=float, default=0.0,
                        help='[video] Starting noise magnitude for temporal scales')
    parser.add_argument('--noise_scale_end_video', type=float, default=0.0,
                        help='[video] Ending noise magnitude for temporal scales')
    parser.add_argument('--noise_stage_video', type=int, default=0,
                        help='[video] Number of temporal scales to apply noise over (0 = disabled)')
    parser.add_argument('--anneal_func_video', type=str, default='linear',
                        choices=['const', 'linear', 'cosine', 'exp', 'A_shape', 'V_shape'],
                        help='[video] Annealing schedule for temporal noise')

    # ---- DiverseVAR: keyframe scale-travel ----
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='[image] Scale-travel passes for keyframe (1 = disabled)')
    parser.add_argument('--encode_stage', type=int, default=0,
                        help='[image] Keyframe scale l (1-based) after which the latent is '
                             're-encoded; scale-travel triggers once scale l has been generated')
    parser.add_argument('--restart_stage', type=int, default=0,
                        help='[image] Number of coarse keyframe scales m (1-based) kept after '
                             're-encoding; generation resumes from scale m+1')

    # ---- DiverseVAR: temporal scale-travel ----
    parser.add_argument('--repeat_num_video', type=int, default=1,
                        help='[video] Scale-travel passes for temporal part (1 = disabled)')
    parser.add_argument('--encode_stage_video', type=int, default=0,
                        help='[video] Temporal scale index (RELATIVE to video start) at which to re-encode')
    parser.add_argument('--restart_stage_video', type=int, default=0,
                        help='[video] Temporal scale index (RELATIVE to video start) to restart from')

    cli = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Fixed model config (checkpoint paths, architecture constants)       #
    # ------------------------------------------------------------------ #
    checkpoints_dir = cli.checkpoints_dir

    args = Args()
    args.pn                    = '0.40M'
    args.fps                   = 16
    args.video_frames          = cli.duration * 16 + 1
    args.model_path            = os.path.join(checkpoints_dir, 'infinitystar_8b_480p_weights')
    args.checkpoint_type       = 'torch_shard'
    args.vae_path              = os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt     = os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.videovae              = 10
    args.model_type            = 'infinity_qwen8b'
    args.text_channels         = 2048
    args.dynamic_scale_schedule= 'infinity_elegant_clip20frames_v2'
    args.bf16                  = 1
    args.use_apg               = 1
    args.use_cfg               = 0
    args.apg_norm_threshold    = 0.05
    args.image_scale_repetition= '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition= '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    args.append_duration2caption= 1
    args.use_two_stage_lfq     = 1
    args.detail_scale_min_tokens= 350
    args.semantic_scales       = 11
    args.max_repeat_times      = 10000

    # ---- override with CLI values ----
    args.cfg            = cli.cfg
    args.tau_image      = cli.tau_image
    args.tau_video      = cli.tau_video
    args.prompt_txt     = cli.prompt_txt

    # DiverseVAR: noise type + shared
    args.noise_type    = cli.noise_type
    args.noise_rescale = cli.noise_rescale
    args.noise_seed    = cli.noise_seed

    # DiverseVAR: keyframe noise  (legacy aliases; explicit *_image overrides follow)
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

    # save_dir is needed by build_parameters for structured save path
    args.save_dir = cli.save_dir

    # ------------------------------------------------------------------ #
    # Build prompt list                                                   #
    # ------------------------------------------------------------------ #
    with open(cli.prompt_txt, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {cli.prompt_txt}")

    # ------------------------------------------------------------------ #
    # Load models (once, shared across all prompts / seeds)              #
    # ------------------------------------------------------------------ #
    pipe = InferencePipe(args)

    # ------------------------------------------------------------------ #
    # Generate + save (perform_inference handles structured path & KV     #
    # cache cleanup internally)                                           #
    # ------------------------------------------------------------------ #
    metadata = {
        "hyperparameters": {
            "cfg":                   args.cfg,
            "tau_image":             args.tau_image,
            "tau_video":             args.tau_video,
            "noise_type":            args.noise_type,
            # "seed":                  args.seed,
            "noise_seed":            args.noise_seed,
            # keyframe noise
            "noise_scale_image":     getattr(args, 'noise_scale_image',     args.noise_scale),
            "noise_stage_image":     getattr(args, 'noise_stage_image',     args.noise_stage),
            "anneal_func_image":     getattr(args, 'anneal_func_image',     args.anneal_func),
            # temporal noise
            "noise_scale_video":     args.noise_scale_video,
            "noise_stage_video":     args.noise_stage_video,
            "anneal_func_video":     args.anneal_func_video,
            # shared noise
            "noise_rescale":         args.noise_rescale,
            # keyframe scale-travel
            "repeat_num":            args.repeat_num,
            "encode_stage":          args.encode_stage,
            "restart_stage":         args.restart_stage,
            # temporal scale-travel (indices relative to video part start)
            "repeat_num_video":      args.repeat_num_video,
            "encode_stage_video":    args.encode_stage_video,
            "restart_stage_video":   args.restart_stage_video,
            "duration":              cli.duration,
            "seed_start":            cli.seed_start,
            "num_seeds":             cli.num_seeds,
        },
        "videos": {},
    }
    inference_times = []
    save_dir_logged = None

    # Global task index across (prompt_idx, seed) pairs, used for sharding
    # work across multiple workers (when launched with --worker_id / --num_workers).
    task_idx = 0

    for prompt_idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        for seed in range(cli.seed_start, cli.seed_start + cli.num_seeds):
            # Shard (prompt_idx, seed) pairs across workers if num_workers > 1
            if cli.num_workers > 1:
                if task_idx % cli.num_workers != cli.worker_id:
                    task_idx += 1
                    continue

            prompt_tag = f"{prompt_idx:0>5}_{seed:0>2}"
            data = {
                'seed':     seed,
                'prompt':   prompt,
                'duration': cli.duration,
            }
            result = perform_inference(pipe, data, args, prompt_tag)

            video_key = f"{prompt[:40]}__seed_{seed:03d}"
            metadata["videos"][video_key] = {
                "prompt":       prompt,
                "seed":         seed,
                "save_path":    result["save_path"],
                "elapsed_time": result["elapsed_time"],
            }
            inference_times.append(result["elapsed_time"])
            save_dir_logged = result["save_dir"]
            # print(f"Video generation done: {result['save_path']}  ({result['elapsed_time']:.1f}s)")

            task_idx += 1

    # ------------------------------------------------------------------ #
    # Save metadata JSON (mirrors Infinity/inference.py)                 #
    # ------------------------------------------------------------------ #
    if save_dir_logged is not None:
        if len(inference_times) > 1:
            metadata["average_inference_time"] = float(np.mean(inference_times[1:]))
        else:
            metadata["average_inference_time"] = float(inference_times[0]) if inference_times else 0.0
        metadata_path = osp.join(save_dir_logged, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
