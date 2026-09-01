import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import time
from pathlib import Path
import json
import gc

from tools.run_infinity import *
from infinity.utils.diverse_var_utils import *

import warnings
warnings.filterwarnings("ignore")

def force_cleanup():
    """Force cleanup of unused tensors"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def build_parameters(args, verbose=False):
    h_div_w = args.h_div_w
    # load scale schedule 
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # load cond-annealing parameters
    # 1. Hyperparameters for noise injection
    if args.noise_type in ("cads", "sos", "prefix", "random"):
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
    save_path = args.save_path + f"/cfg={args.cfg}_tau={args.tau}_top-p={args.top_p}_top-k={args.top_k}" + f"/{prompt_name}/{args.noise_type}/{args.anneal_func}"
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
    metadata = build_parameters(args)
    save_path = metadata['save_path']

    # 2. prepare prompt 
    with open(args.prompt_txt, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if Path(f"{save_path}/metadata.json").exists():
        with open(f"{save_path}/metadata.json", "r") as f:
            metadata = json.load(f)
        metadata['hyperparameters']['scale_schedule'] = tuple(tuple(t) for t in metadata['hyperparameters']['scale_schedule'])
        # if len(metadata["images"]) == len(prompts) * args.imgs_per_prompt:
        #     print(f"[INFO] metadata.json already exists, skipping inference")
        #     return metadata
        # else:
        #     print(f"[INFO] metadata.json already exists, but some images are missing, continuing inference")
    
    # save metadata
    if args.save_indv:
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    # 2. build models 
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    idx_list = []
    total_summed_codes = []
    inference_times = []
    # inference
    if args.img_idx is not None:
        pbar = tqdm(total=len(prompts), desc="Inference")
    else:
        pbar = tqdm(total=len(prompts) * args.imgs_per_prompt, desc="Inference")
    for img_idx, prompt in enumerate(prompts): 

        if args.img_idx is not None and img_idx % args.parallel_num != args.img_idx:
            continue

        # force_cleanup()

        for j in range(args.imgs_per_prompt):

            if args.img_idx is not None:
                j = args.img_idx

            seed = img_idx * args.imgs_per_prompt + j

            img_name = f"{img_idx:0>5}_{j:0>2}"
            if Path(f"{save_path}/{img_name}.{args.image_exts}").exists():
                metadata["images"][img_name] = {
                    "prompt": prompt,
                    "seed": seed,
                }
                pbar.update(1)
                continue

            idx_Bld_list, summed_codes_list, cost = generate_image_codes(
                infinity_test=infinity, 
                vae=vae, 
                text_tokenizer=text_tokenizer, 
                text_encoder=text_encoder, 
                prompt=prompt, 
                top_k=args.top_k,
                top_p=args.top_p,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                g_seed=seed,
                sampling_per_bits=args.sampling_per_bits,

                hyperparameters=metadata['hyperparameters'],
            )
            if args.save_indv: 
                images = idx_to_images(idx_Bld_list, vae, downscale=args.down_res, all_stages=False)
                images[0].save(f"{save_path}/{img_name}.{args.image_exts}")
                metadata["images"][img_name] = {
                    "prompt": prompt,
                    "seed": seed,
                }
            inference_times.append(cost)
            # idx_list.append(idx_Bld_list)
            if args.save_traj or args.save_total:
                total_summed_codes.append(summed_codes_list.squeeze(1))
            pbar.update(1)

    metadata["average_inference_time"] = (
        float(np.mean(inference_times[1:]))
        if len(inference_times) > 1
        else float(inference_times[0]) if inference_times else 0.0
    )

    # # save combined images
    # if args.save_traj or args.save_total:
    #     total_idx = []
    #     for i in range(len(idx_list[0])):
    #         idxes = []
    #         for j in range(len(idx_list)):
    #             idxes.append(idx_list[j][i])
    #         total_idx.append(torch.cat(idxes, dim=0))

    if args.save_traj:
        image = summed_codes_to_images(total_summed_codes, vae, downscale=8, all_stages=True)
        image.save(f"{save_path}/summed_traj.jpg")
    if args.save_total: 
        images = summed_codes_to_images(total_summed_codes, vae, downscale=8, all_stages=False)
        rows = [] 
        for r in range(args.row * args.batch_size):
            row_images = images[r * args.col:(r + 1) * args.col]
            row = stack_images_horizontally(row_images)
            rows.append(row)
        stack_images_vertically(rows, save_path=f"{save_path}/summed_result.jpg")

    # save metadata
    if args.save_indv:
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


    del text_tokenizer, text_encoder, vae, infinity

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
    from infinity.utils.diverse_var_utils import build_args
    args = build_args()

    if args.network_type == "infinity_2b":
        args.pn = '1M'
        args.vae_type = 32
        args.model_type = 'infinity_2b'
        args.checkpoint_type = 'torch'

    elif args.network_type == "infinity_8b_512x512":
        args.pn = '0.25M'
        args.vae_type = 14
        args.model_type = 'infinity_8b'
        args.checkpoint_type = 'torch_shard'
        args.apply_spatial_patchify = 1

    elif args.network_type == "infinity_8b":
        raise NotImplementedError("infinity_8b is not supported yet")

    if not args.model_path or not args.vae_path:
        raise ValueError("--model_path and --vae_path are required; see README.md")
    if not args.prompt_txt:
        raise ValueError("--prompt_txt is required; see README.md")

    

    metadata = inference(args)

    if args.eval:
        evaluation(metadata, args)
