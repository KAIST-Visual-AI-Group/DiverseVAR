import argparse
import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

from models.pipeline import SwittiPipeline
from utils_local.diverse_var_utils import *

import warnings
warnings.filterwarnings("ignore")

def build_parameters(args, verbose=False):

    if args.model_path.split("/")[-1] == "Switti-1024":
        scale_schedule = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64)
    else:
        raise ValueError(f"Invalid model path: {args.model_path}")

    # load cond-annealing parameters
    # 1. Hyperparameters for noise injection
    if args.noise_type in ("cond_bd", "context", "prefix", "random", "both"):
        if args.noise_stage <= 1:
            noise_scales = [0.0] * len(scale_schedule)
        elif args.anneal_func == 'linear':
            noise_scales = list(np.linspace(args.noise_scale, 0.0, args.noise_stage)) + [0.0] * (len(scale_schedule) - args.noise_stage)
        elif args.anneal_func == 'cosine':
            noise_scales = list(cosine_annealing(args.noise_scale, 0.0, args.noise_stage)) + [0.0] * (len(scale_schedule) - args.noise_stage)
        elif args.anneal_func == 'const':
            noise_scales = [args.noise_scale] * (args.noise_stage - 1) + [0.0] * (len(scale_schedule) - args.noise_stage + 1)
        else:
            raise ValueError(
                f"--anneal_func {args.anneal_func} is not implemented for --noise_type "
                f"{args.noise_type}; use one of linear / cosine / const."
            )
        if verbose:
            print(f"[args.noise_scale: {args.noise_scale}][args.noise_stage: {args.noise_stage}] noise_scales: {noise_scales}")
    else: 
        noise_scales = [0.0] * len(scale_schedule)

    # 2. Hyperparameters for cfg / tau annealing 
    # if args.noise_type == "cfg":
    if args.noise_type == "cfg_zero" or args.noise_type == "cfg_one":
        assert args.noise_scale == 1.0, "noise_scale must be 1.0 for cfg_zero or cfg_one"
        # assert args.noise_type == "cfg_zero", "cfg_zero should be used for (1 + w) * uncond - w * cond"
        cfg_list = [args.cfg] * len(scale_schedule)
        if args.noise_type == "cfg_zero":
            start_cfg = args.cfg * (1.0 - args.noise_scale)
        elif args.noise_type == "cfg_one":
            start_cfg = args.cfg * (1.0 - args.noise_scale) + 1.0
        if args.noise_stage > 1 and args.anneal_func == 'linear':
            cfg_list = list(np.linspace(start_cfg, args.cfg, args.noise_stage)) + [args.cfg] * (len(scale_schedule) - args.noise_stage)
        elif args.noise_stage > 1 and args.anneal_func == 'cosine':
            cfg_list = list(cosine_annealing(start_cfg, args.cfg, args.noise_stage)) + [args.cfg] * (len(scale_schedule) - args.noise_stage)

        elif args.noise_stage > 1 and args.anneal_func == "invlinear":
            cfg_list = list(np.linspace(args.cfg, start_cfg, args.noise_stage)) + [args.cfg] * (len(scale_schedule) - args.noise_stage)
        elif args.noise_stage > 1 and args.anneal_func == "invcosine":
            cfg_list = list(cosine_annealing(args.cfg, start_cfg, args.noise_stage)) + [args.cfg] * (len(scale_schedule) - args.noise_stage)

        elif args.noise_stage > 1 and args.anneal_func == "A_shape":
            cfg_list = list(a_shape_annealing(start_cfg, args.cfg, args.noise_stage - 1)) + [args.cfg] * (len(scale_schedule) - args.noise_stage + 1)
        elif args.noise_stage > 1 and args.anneal_func == "V_shape":
            cfg_list = list(v_shape_annealing(start_cfg, args.cfg, args.noise_stage - 1)) + [args.cfg] * (len(scale_schedule) - args.noise_stage + 1)
        
        elif args.noise_stage > 1 and args.anneal_func == "static":
            cfg_list = [0.5 * args.cfg] * (args.noise_stage - 1) + [args.cfg] * (len(scale_schedule) - args.noise_stage + 1)

        elif args.noise_stage > 1 and args.anneal_func == 'const':
            cfg_list = [start_cfg] * (args.noise_stage - 1) + [args.cfg] * (len(scale_schedule) - args.noise_stage + 1)
        elif args.anneal_func == 'const_reverse':
            cfg_list = [args.cfg] * (args.noise_stage - 1) + [start_cfg] * (len(scale_schedule) - args.noise_stage + 1)
        if verbose:
            print(f"[args.cfg: {args.cfg}][args.noise_stage: {args.noise_stage}] cfg_list: {cfg_list}")
    else:
        cfg_list = [args.cfg] * len(scale_schedule)
    
    # Not implemented yet 
    if args.noise_type == "tau":
        assert args.noise_scale == 1.0, "noise_scale must be 1.0 for tau"
        tau_list = [args.tau] * len(scale_schedule)
        start_tau = 10.0
        if args.noise_stage > 1 and args.anneal_func == 'linear':
            tau_list = list(np.linspace(start_tau, args.tau, args.noise_stage)) + [args.tau] * (len(scale_schedule) - args.noise_stage)
        elif args.noise_stage > 1 and args.anneal_func == 'cosine':
            tau_list = list(cosine_annealing(start_tau, args.tau, args.noise_stage)) + [args.tau] * (len(scale_schedule) - args.noise_stage)

        elif args.noise_stage > 1 and args.anneal_func == "invlinear":
            tau_list = list(np.linspace(args.tau, start_tau, args.noise_stage)) + [args.tau] * (len(scale_schedule) - args.noise_stage)
        elif args.noise_stage > 1 and args.anneal_func == "invcosine":
            tau_list = list(cosine_annealing(args.tau, start_tau, args.noise_stage)) + [args.tau] * (len(scale_schedule) - args.noise_stage)

        elif args.noise_stage > 1 and args.anneal_func == "A_shape":
            tau_list = list(a_shape_annealing(start_tau, args.tau, args.noise_stage - 1)) + [args.tau] * (len(scale_schedule) - args.noise_stage + 1)
        elif args.noise_stage > 1 and args.anneal_func == "V_shape":
            tau_list = list(v_shape_annealing(start_tau, args.tau, args.noise_stage - 1)) + [args.tau] * (len(scale_schedule) - args.noise_stage + 1)
        
        elif args.noise_stage > 1 and args.anneal_func == "static":
            tau_list = [0.5 * args.tau] * (args.noise_stage - 1) + [args.tau] * (len(scale_schedule) - args.noise_stage + 1)

        elif args.noise_stage > 1 and args.anneal_func == 'const':
            tau_list = [start_tau] * (args.noise_stage - 1) + [args.tau] * (len(scale_schedule) - args.noise_stage + 1)
        elif args.anneal_func == 'const_reverse':
            tau_list = [args.tau] * (args.noise_stage - 1) + [start_tau] * (len(scale_schedule) - args.noise_stage + 1)
        if verbose:
            print(f"[args.tau: {args.tau}][args.noise_stage: {args.noise_stage}] tau_list: {tau_list}")
    else:
        tau_list = [args.tau] * len(scale_schedule)

    cond_anneal_params = {
        "noise_type": args.noise_type,
        "noise_scales": noise_scales,
        "noise_stage": args.noise_stage,
        "noise_rescale": args.noise_rescale,
        "noise_seed": args.noise_seed,

        "cfg_list": cfg_list,
        "tau_list": tau_list,
    }

    if verbose:
        print(f"[DEBUG] noise_type: {args.noise_type}, anneal_func: {args.anneal_func}")
        print(f"[DEBUG] noise_scale: {args.noise_scale}, noise_stage: {args.noise_stage}")
        print(f"[DEBUG] cfg: {args.cfg}, cfg_list: {cfg_list}")
        print(f"[DEBUG] tau: {args.tau}, tau_list: {tau_list}")
        print(f"[DEBUG] noise_scales: {noise_scales}")
    
    # load scale-travel parameters
    scale_travel_params = {
        "repeat_num": args.repeat_num,
        "encode_stage": args.encode_stage,
        "restart_stage": args.restart_stage,
        "cfg_refine": [args.cfg] * len(scale_schedule),
    }

    # load other parameters
    prompt_name = Path(args.prompt_txt).stem

    if args.cfg == 7.0 and args.tau == 1.0 and args.top_k == 400 and args.top_p == 0.95:
        save_path = args.save_path
    else:
        save_path = args.save_path + f"/{args.cfg}_{args.tau}_{args.top_k}_{args.top_p}"

    save_path = save_path + f"/{prompt_name}/{args.noise_type}/{args.anneal_func}"
    save_path += f"/{args.noise_scale}_{args.noise_stage}"
    if args.repeat_num <= 1: 
        save_path += "/no_scale_travel"
    else:
        save_path += f"/{args.repeat_num}_{args.encode_stage}_{args.restart_stage}"
    os.makedirs(save_path, exist_ok=True)
    if verbose:
        print(f"[DEBUG] save_path: {save_path}")

    # save parameters 
    metadata = {
        "save_path": save_path,
        "hyperparameters": {
            "args": vars(args),
            "scale_schedule": scale_schedule,
            "cond_anneal_params": cond_anneal_params,
            "scale_travel_params": scale_travel_params,
        },
        "images": {

        },
    }

    return metadata

def inference(args):

    # 1. build parameters
    metadata = build_parameters(args, verbose=args.verbose)
    save_path = metadata['save_path']
    print(f"[Inference] save_path: {save_path}")

    # 2. prepare prompt 
    with open(args.prompt_txt, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if Path(f"{save_path}/metadata.json").exists():
        with open(f"{save_path}/metadata.json", "r") as f:
            metadata = json.load(f)
        if len(metadata["images"]) == len(prompts) * args.imgs_per_prompt:
            print(f"[INFO] metadata.json already exists, skipping inference")
            return metadata
        else:
            print(f"[INFO] metadata.json already exists, but some images are missing, continuing inference")
    
    
    ca_params = metadata['hyperparameters']['cond_anneal_params']
    st_params = metadata['hyperparameters']['scale_travel_params']
    
    # 3. build models 
    device = 'cuda:0'

    pipe = SwittiPipeline.from_pretrained(args.model_path, device=device, torch_dtype=torch.bfloat16)
    print(f"Number of parameters (switti): {sum(p.numel() for p in pipe.switti.parameters())}")




    inference_times = []
    images_list = []
    f_hats_list = []
    # inference
    pbar = tqdm(total=len(prompts) * args.imgs_per_prompt, desc="Switti Inference")
    # prompts = prompts * args.imgs_per_prompt
    prompts = [p for p in prompts for _ in range(args.imgs_per_prompt)]
    prompts_batch, img_names_batch = [], []
    for img_idx, prompt in enumerate(prompts): 
        prompts_batch.append(prompt)
        i = img_idx // args.imgs_per_prompt
        j = img_idx % args.imgs_per_prompt
        img_names_batch.append(f"{i:0>5}_{j:0>2}")

        if len(prompts_batch) == args.batch_size or img_idx == len(prompts) - 1:

            # check if all images exist
            all_exist = True
            for img_name in img_names_batch:
                if not os.path.exists(f"{save_path}/{img_name}.{args.image_exts}"):
                    all_exist = False
                    break
            if all_exist:
                # print(f"[INFO] {img_names_batch} already exists, skipping inference")
                for i, img_name in enumerate(img_names_batch):
                    metadata["images"][img_name] = {
                        "prompt": prompts_batch[i],
                        "seed": img_idx,
                    }
                pbar.update(len(prompts_batch))
                prompts_batch = []
                img_names_batch = []
                continue

            start_time = time.time()
            seed = img_idx
            
            images, f_hats = pipe(prompts_batch,
                cfg_list=ca_params['cfg_list'],
                tau_list=ca_params['tau_list'],
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=args.more_smooth,
                cfg_anneal=args.cfg_anneal,
                return_pil=True,
                smooth_start_si=args.smooth_start_si,
                turn_on_cfg_start_si=args.turn_on_cfg_start_si,
                turn_off_cfg_start_si=args.turn_off_cfg_start_si,
                last_scale_temp=args.last_scale_temp,
                seed=seed,
                return_f_hats=args.save_traj,

                # [DiverseVAR] 1. cond-annealing params 
                noise_type = ca_params['noise_type'],
                noise_scales = ca_params['noise_scales'],
                noise_rescale = ca_params['noise_rescale'],
                noise_seed = ca_params['noise_seed'],

                # [DiverseVAR] 2. Scale-Travel params 
                repeat_num = st_params['repeat_num'],
                encode_stage = st_params['encode_stage'],
                restart_stage = st_params['restart_stage'],
                cfg_refine = st_params['cfg_refine'],
            )
            if args.save_indv: 
                for i, img in enumerate(images):
                    img.save(f"{save_path}/{img_names_batch[i]}.{args.image_exts}")
                    metadata["images"][img_names_batch[i]] = {
                        "prompt": prompts_batch[i],
                        "seed": seed,
                    }
            inference_times.append(time.time() - start_time)
            if args.save_total:
                images_list += images
            if args.save_traj:  
                f_hats_list.append(f_hats)
            pbar.update(len(prompts_batch))
            prompts_batch = []
            img_names_batch = []

    metadata["average_inference_time"] = (
        float(np.mean(inference_times[1:]))
        if len(inference_times) > 1
        else float(inference_times[0]) if inference_times else 0.0
    )

    # save combined images
    if args.save_total:
        tot_images = []
        for i in range(args.row):
            row_images = []
            for j in range(args.col):
                row_images.append(images_list[i * args.col + j].resize((1024 // args.down_res, 1024 // args.down_res)))
            row = stack_images_horizontally(row_images)
            tot_images.append(row)
        stack_images_vertically(tot_images, save_path=f"{save_path}/summed_result.jpg")

    if args.save_traj:
        tot_images = []
        for si in range(len(f_hats_list[0])):
            traj_images = []
            for f_hat in f_hats_list:
                h, w = f_hat[si][0].size
                traj_images.append(f_hat[si][i].resize((h // args.down_res, w // args.down_res)) for i in range(len(f_hat[si])))
            row = stack_images_horizontally(traj_images)
            tot_images.append(row)
        stack_images_vertically(tot_images, save_path=f"{save_path}/traj.jpg")

        # raise NotImplementedError("save_traj is not implemented yet")

    # save metadata
    if args.save_indv:
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


    del pipe

    return metadata


def evaluation(metadata, args):
    """Run the shared evaluator (`evaluate.py`) on the folder just generated."""
    from evaluate import evaluate_folder

    metrics = [m.strip() for m in args.eval_metrics.split(",") if m.strip()]
    return evaluate_folder(
        metadata['save_path'],
        metrics,
        image_exts=args.image_exts,
    )





if __name__ == "__main__":
    from utils_local.diverse_var_utils import build_args
    args = build_args()

    if not args.prompt_txt:
        raise ValueError("--prompt_txt is required; see README.md")

    metadata = inference(args)

    if args.eval:
        evaluation(metadata, args)
