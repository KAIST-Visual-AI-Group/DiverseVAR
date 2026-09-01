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
from infinity.utils.arg_util import Args


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


def perform_inference(pipe, data, args):
    
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=5
    num_frames=81

    # If an image_path is provided, perform image-to-video generation.
    image_path = data.get("image_path", None)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)    
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    if image_path is not None:
        ref_image = [cv2.imread(image_path)[:,:,::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0) # [t,3,h,w]
        ref_img_bcthw = ref_img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(pipe.vae, ref_img_bcthw.cuda(), vae_features=None, self_correction=pipe.self_correction, args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak=len(scale_schedule)//2

    generated_image_list = []
    negative_prompt=''
    prompt = f'{prompt}, Close-up on big objects, emphasize scale and detail'
    negative_prompt = ""
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt
    
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
            cfg_list=args.cfg, 
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
            noise_list=None,
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        print(generated_image.shape)
        generated_image_list.append(generated_image)
            
    generated_image = torch.cat(generated_image_list, 2)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    
    return {
            "output": generated_image.cpu().numpy(),
            "elapsed_time": elapsed_time,
        }


if __name__ == '__main__':
    checkpoints_dir = os.environ.get('INFINITYSTAR_CHECKPOINTS', 'pretrained/InfinityStar')
    

    # infer args
    args = Args()
    args.pn='0.90M'
    args.fps=16
    args.video_frames=81
    args.model_path=os.path.join(checkpoints_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type='torch_shard' # omnistore
    args.vae_path=os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt=os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type='infinity_qwen8b'
    args.text_channels=2048
    args.dynamic_scale_schedule='infinity_elegant_clip20frames_v2'
    args.bf16=1
    args.use_apg=1
    args.use_cfg=0
    args.cfg=34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold=0.05
    args.image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption=1
    args.use_two_stage_lfq=1
    args.detail_scale_min_tokens=750
    args.semantic_scales=12
    args.max_repeat_times=10000

    # load models
    pipe = InferencePipe(args)

    
    prompt = "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic"
    # image_path = 'assets/reference_image.webp'  # Remove this for Text-to-Video (T2V) generation
    data = {
        'seed': 41,
        # 'image_path': image_path,
        'prompt': prompt,
    }

    output_dict = perform_inference(pipe, data, args)
    save_dir = 'output'
    gen_video_path = osp.join(os.path.join(save_dir, 'gen_videos'), f'demo.mp4')
    save_video(output_dict['output'], fps=args.fps, save_filepath=gen_video_path)
            
    print(f"Video genernation done: {gen_video_path=}")
