import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


###############################################
# [DiverseVAR] Autoregressive Inference utils #
###############################################

# def make_scalewise_lower_triangle_mask(k: int):
#     # Compute token indices per level
#     scale_schedule = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
#     level_sizes = [(scale_schedule[i]) ** 2 for i in range(k + 1)]  # [1, 4, 9, ...]
#     level_sizes[0] = 300
#     level_ends = torch.cumsum(torch.tensor(level_sizes), dim=0)  # [1, 5, 14, ...]
#     total_len = level_ends[-1].item()

#     mask = torch.full((total_len, total_len), float('-inf'))  # default: -inf (invisible)

#     # Set visible regions for each level
#     for i in range(len(level_ends)):
#         start = 0 if i == 0 else level_ends[i - 1].item()
#         end = level_ends[i].item()
#         # tokens in this level (end-start) can attend to everything up to `end`
#         mask[start:end, :end] = 0

#     # [1, 1, L, L] for transformer
#     return mask.unsqueeze(0).unsqueeze(0)

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
        sigma = torch.randn(tensor.shape, generator=noise_generator).to(tensor.device).to(tensor.dtype)
        tensor = np.sqrt(1 - noise_scale) * tensor + sigma * np.sqrt(noise_scale)
        if rescale > 0:
            tensor_scaled = (tensor - torch.mean(tensor)) / torch.std(tensor) * std + mean
            tensor = rescale * tensor_scaled + (1 - rescale) * tensor

    return tensor


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



def tensor_to_image(tensor):
    tensor = tensor.mul_(255).cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1).astype(np.uint8)
    return Image.fromarray(tensor[0])
    
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
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to Switti model.",
        default="yresearch/Switti-1024",
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=False,
    )
    parser.add_argument('--verbose',        default=False, type=bool)

    # ----------- Hyperparameters for inference ----------- #
    parser.add_argument('--cfg',            default=6.0, type=float)
    parser.add_argument('--tau',            default=1.0, type=float)
    parser.add_argument('--top_k',          default=400, type=int)
    parser.add_argument('--top_p',          default=0.95, type=float)
    parser.add_argument('--cfg_anneal',     default=0, type=int)

    parser.add_argument('--smooth_start_si', default=2, type=int)
    parser.add_argument('--turn_on_cfg_start_si', default=2, type=int)
    parser.add_argument('--turn_off_cfg_start_si', default=11, type=int)
    parser.add_argument('--last_scale_temp', default=0.1, type=float)

    parser.add_argument('--batch_size',     default=1, type=int)
    parser.add_argument('--col',            default=1, type=int)
    parser.add_argument('--row',            default=1, type=int)
    parser.add_argument('--imgs_per_prompt', default=1, type=int)
    parser.add_argument('--prompt_num',     default=None, type=int)

    parser.add_argument('--save_path',      default='.', type=str)
    parser.add_argument('--save_traj',      default=0, type=int)
    parser.add_argument('--save_indv',      default=0, type=int)
    parser.add_argument('--save_total',     default=0, type=int)
    parser.add_argument('--down_res',       default=1, type=int)

    # ----------- Hyperparameters for noise injection ----------- #
    parser.add_argument('--noise_type',     default='none', type=str, 
                        choices=['none', 'cfg_zero', 'cfg_one', 'tau', 'cond_bd', 'context', 'prefix', 'random', 'both'])
    
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
    parser.add_argument('--image_exts', default='jpg', type=str)
    args = parser.parse_args()
    return args