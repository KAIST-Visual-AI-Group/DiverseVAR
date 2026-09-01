import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


###############################################
# [DiverseVAR] Autoregressive Inference utils #
###############################################

def inject_noise_with_rescale(tensor, noise_scale, rescale=1.0, noise_generator=None):
    """
    Inject noise to the tensor and rescale it.

    params:
        tensor: torch.Tensor, the tensor to inject noise to
        noise_scale: float, the scale of the noise
        rescale: float, the rescale factor
        noise_generator: torch.Generator, the generator to use for noise
    return:
        tensor: torch.Tensor, the tensor with noise injected
    """
    tensor = tensor.clone()
    mean, std = torch.mean(tensor), torch.std(tensor)
    if noise_scale > 0:
        sigma = torch.randn(tensor.shape, generator=noise_generator).to(tensor.device)
        tensor = np.sqrt(1 - noise_scale) * tensor + sigma * np.sqrt(noise_scale)
        if rescale > 0:
            tensor_scaled = (tensor - torch.mean(tensor)) / torch.std(tensor) * std + mean
            tensor = rescale * tensor_scaled + (1 - rescale) * tensor

    return tensor


def make_scalewise_lower_triangle_mask(scale_schedule, k: int, device=None, dtype=None):
    """
    scale_schedule: List[Tuple[int,int,int]]  # (pt, h, w) per stage
    k: current stage index (0-based), build mask for stages [0..k]
    returns: [1,1,L,L] with 0 for visible, -inf for masked
    """
    assert isinstance(scale_schedule, (list, tuple)) and len(scale_schedule) >= k + 1
    level_sizes = [int(pt * h * w) for (pt, h, w) in scale_schedule[:k+1]]
    level_ends = torch.tensor(level_sizes, device=device).cumsum(0)
    total_len = int(level_ends[-1].item())

    mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype)

    prev_end = 0
    for end in level_ends.tolist():
        # tokens in [prev_end:end) can attend to [0:end)
        mask[prev_end:end, :end] = 0
        prev_end = end

    return mask.unsqueeze(0).unsqueeze(0)


##########################################
# [DiverseVAR] Image Visualization utils #
##########################################

def stack_images_vertically(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.convert("RGB").save(save_path)
    return new_im.convert("RGB")

def stack_images_horizontally(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.convert("RGB").save(save_path)
    return new_im.convert("RGB")
    
def idx_to_images(idxes, vae, downscale=1, all_stages=False):

    def downscale_image(image, scale):
        if scale == 1:
            return image
        else:
            return image.resize((int(image.width // scale), int(image.height // scale)), Image.LANCZOS)

    # intp_size = (1, 64, 64)
    intp_size = idxes[-1].shape[1:-1]

    codes = []
    for i in range(len(idxes)):
        codes.append(vae.quantizer.lfq.indices_to_codes(idxes[i], label_type='bit_label'))
    
    summed_codes_list = [] 
    summed_codes = 0
    for i in range(len(codes)):
        summed_codes += F.interpolate(codes[i], size=intp_size, mode=vae.quantizer.z_interplote_up)
        summed_codes_list.append(summed_codes.clone())
    
    summed_codes = torch.stack(summed_codes_list, dim=0)
    
    images = [] 
    if all_stages: 
        for i in range(summed_codes.shape[1]):
            img = []
            for j in range(summed_codes.shape[0]):
                img.append(vae.decode(summed_codes[j:j+1, i].squeeze(-3)))
            img = torch.stack(img).squeeze()
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            imgs = [Image.fromarray(img[j].cpu().numpy().astype(np.uint8)[:, :, ::-1]) for j in range(len(img))]
            imgs = [downscale_image(img, downscale) for img in imgs]
            images.append(stack_images_horizontally(imgs))

        return stack_images_vertically(images)

    else:
        for i in range(summed_codes.shape[1]):
            img = vae.decode(summed_codes[-1:, i].squeeze(-3))
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            img = Image.fromarray(img.squeeze().cpu().numpy().astype(np.uint8)[:, :, ::-1])
            img = downscale_image(img, downscale)
            images.append(img)
        return images
    
def summed_codes_to_images(summed_codes_list, vae, downscale=1, all_stages=False, save_all_traj=False):

    def downscale_image(image, scale):
        if scale == 1:
            return image
        else:
            return image.resize((int(image.width // scale), int(image.height // scale)), Image.LANCZOS)

    summed_codes = torch.stack(summed_codes_list, dim=0)    # [N_sample, N_stage, 32, 64, 64]
    # print(f"total summed_codes shape: {summed_codes.shape}")
    images = [] 
    if all_stages and save_all_traj == False: 
        for i in range(summed_codes.shape[1]):
            img = []
            for j in range(summed_codes.shape[0]):
                # print(f"summed_codes shape: {summed_codes[j:j+1, i].squeeze(-3).shape}")
                img.append(vae.decode(summed_codes[j:j+1, i].squeeze(-3)))
            img = torch.stack(img).squeeze()
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            imgs = [Image.fromarray(img[j].cpu().numpy().astype(np.uint8)[:, :, ::-1]) for j in range(len(img))]
            imgs = [downscale_image(img, downscale) for img in imgs]
            images.append(stack_images_vertically(imgs))

        return stack_images_horizontally(images).convert("RGB")
    
    elif all_stages and save_all_traj: 
        for i in range(summed_codes.shape[1]):
            img = []
            for j in range(summed_codes.shape[0]):
                # print(f"summed_codes shape: {summed_codes[j:j+1, i].squeeze(-3).shape}")
                img.append(vae.decode(summed_codes[j:j+1, i].squeeze(-3)))
            img = torch.stack(img).squeeze()
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            imgs = [Image.fromarray(img[j].cpu().numpy().astype(np.uint8)[:, :, ::-1]) for j in range(len(img))]
            imgs = [downscale_image(img, downscale) for img in imgs]
            for j in range(len(imgs)):
                images.append(imgs[j])

        return images

    else:
        for i in range(summed_codes.shape[0]):
            img = vae.decode(summed_codes[i, -1:].squeeze(-3))
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            img = Image.fromarray(img.squeeze().cpu().numpy().astype(np.uint8)[:, :, ::-1])
            img = downscale_image(img, downscale).convert("RGB")
            images.append(img)
        return images
    
def image_to_tensor(images):
    tensors = []
    for img in images:
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensors.append(img)
    return torch.stack(tensors)


##########################################
# [DiverseVAR] Condition-Annealing utils #
##########################################

def cosine_annealing(start: float, end: float, steps: int):
    """
    Generate a cosine annealing array that smoothly transitions from start to end value.
    
    Args:
        start (float): Initial value
        end (float): Final value
        steps (int): Number of steps in the sequence
        
    Returns:
        np.ndarray: Array of values following cosine annealing schedule
    """
    x = np.linspace(0, np.pi, steps)
    return end + (start - end) * (1 + np.cos(x)) / 2


def a_shape_annealing(start: float, end: float, steps: int):
    up = np.linspace(start, end, 1000)
    down = np.linspace(end, start, 1000)
    total = np.concatenate([up, down])
    idx = [2000 // (steps - 1) * i for i in range(steps - 1)] + [1999]
    return total[idx]

def v_shape_annealing(start: float, end: float, steps: int):
    up = np.linspace(start, end, 1000)
    down = np.linspace(end, start, 1000)
    total = np.concatenate([down, up])
    idx = [2000 // (steps - 1) * i for i in range(steps - 1)] + [1999]
    return total[idx]

def exponential_annealing(start, end, steps, epsilon=1e-2):
    x = np.linspace(0.0, -np.log(start * epsilon), steps)
    return [np.exp(-x_i) * start for x_i in x]

def linear_annealing(start, end, steps):
    return [
        start + (end - start) * (i / (steps - 1))
        for i in range(steps)
    ]



##########################################
# [DiverseVAR] Inference Argument Parser #
##########################################

import argparse
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pn', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--cfg_insertion_layer', default=0, type=int)
    parser.add_argument('--vae_type', type=int)
    parser.add_argument('--vae_path', type=str)
    parser.add_argument('--add_lvl_embeding_only_first_block', default=1, type=int)
    parser.add_argument('--use_bit_label', default=1, type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--rope2d_each_sa_layer', default=1, type=int)
    parser.add_argument('--rope2d_normalized_by_hw', default=2, type=int)
    parser.add_argument('--use_scale_schedule_embedding', default=0, type=int)
    parser.add_argument('--sampling_per_bits', default=1, type=int)
    parser.add_argument('--text_encoder_ckpt', default='google/flan-t5-xl', type=str)
    parser.add_argument('--text_channels', default=2048, type=int)
    parser.add_argument('--apply_spatial_patchify', default=0, type=int)
    parser.add_argument('--h_div_w_template', default=1.0, type=float)
    parser.add_argument('--use_flex_attn', default=0, type=int)
    parser.add_argument('--cache_dir', default='.cache', type=str)
    parser.add_argument('--checkpoint_type', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bf16', default=1, type=int)
    parser.add_argument('--save_file', default='tmp.jpg', type=str)
    parser.add_argument('--enable_model_cache', default=True, action='store_true')

    parser.add_argument('--network_type', default='infinity_2b', type=str)
    parser.add_argument('--h_div_w', default=1.0, type=float)

    # ----------- Hyperparameters for inference ----------- #
    parser.add_argument('--cfg',            default=3.0, type=float)
    parser.add_argument('--tau',            default=1.0, type=float)
    parser.add_argument('--top_k',          default=900, type=int)
    parser.add_argument('--top_p',          default=0.97, type=float)

    parser.add_argument('--gt_leak',        default=-1, type=int)

    parser.add_argument('--batch_size',     default=1, type=int)
    parser.add_argument('--col',            default=1, type=int)
    parser.add_argument('--row',            default=1, type=int)
    parser.add_argument('--imgs_per_prompt', default=1, type=int)
    parser.add_argument('--prompt_num',     default=None, type=int)
    parser.add_argument('--img_idx',        default=None, type=int)
    parser.add_argument('--parallel_num',   default=1, type=int)

    parser.add_argument('--save_path',      default='.', type=str)
    parser.add_argument('--save_traj',      default=0, type=int)
    parser.add_argument('--save_indv',      default=0, type=int)
    parser.add_argument('--save_total',     default=0, type=int)
    parser.add_argument('--down_res',       default=1, type=int)

    # ----------- Hyperparameters for noise injection ----------- #
    parser.add_argument('--noise_type',     default='none', type=str, 
                        choices=['none', 'cfg_zero', 'cfg_one', 'tau', 'cads', 'sos', 'prefix', 'random'])
    
    parser.add_argument('--noise_seed',     default=730,    type=int)

    parser.add_argument('--noise_scale',    default=0.0,    type=float)
    parser.add_argument('--noise_rescale',  default=0.0,    type=float)
    parser.add_argument('--noise_stage',    default=1,      type=int)
    parser.add_argument('--anneal_func',    default='linear', type=str, 
                        choices=['const', 'linear', 'cosine', 'exp', 'invlinear', 
                                 'invcosine', 'const_reverse', 'A_shape', 'V_shape', 'static'])

    # ----------- Hyperparameters for refinement ----------- #
    parser.add_argument('--repeat_num',     default=1,      type=int)
    parser.add_argument('--encode_stage',   default=1,      type=int)
    parser.add_argument('--restart_stage',  default=1,      type=int)

    parser.add_argument('--prompt_txt',     default='', type=str)

    # ----------- Hyperparameters for noise annealing ----------- #
    parser.add_argument('--noise_scale_end', default=0.0, type=float)

    # ----------- Hyperparameters for evaluation ----------- #
    # `--eval 1` hands the generated folder to the shared evaluator in evaluate.py.
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--eval_metrics', default='LPIPS,Vendi,ImageReward,FID', type=str)
    parser.add_argument('--image_exts', default='png', type=str)
    args = parser.parse_args()
    return args




