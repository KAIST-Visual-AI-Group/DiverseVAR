import torch
import numpy as np
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
        mask[prev_end:end, :end] = 0
        prev_end = end

    return mask.unsqueeze(0).unsqueeze(0)


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


#########################################################
# [DiverseVAR] Scale-Travel re-quantization utility     #
#########################################################

@torch.no_grad()
def reencode_for_scale_travel(summed_code, vae, vae_scale_schedule, restart_stage, args, target_pn, interpolate_fn, initial_noise=None):
    """
    Re-encode accumulated latent codes to obtain cleaner per-scale bit indices
    for Scale-Travel (refinement) restart.

    Args:
        summed_code: [B, C, t, h, w] accumulated latent at target resolution
        vae: video VAE with quantizer
        vae_scale_schedule: list of (pt, ph, pw) tuples
        restart_stage: int, number of scales to re-encode (restart from this stage)
        args: inference args (used for use_two_stage_lfq)
        target_pn: target resolution tuple (t, h, w)
        interpolate_fn: the interpolate function from infinity_elegant
        initial_noise: optional initial noise for the first scale (used as starting cum_var_input)

    Returns:
        new_bit_indices: list of bit index tensors per scale [0..restart_stage-1]
        new_summed_code: re-encoded accumulated codes [B, C, t, h, w]
    """
    with torch.amp.autocast('cuda', enabled=False):
        target = summed_code.float()
        if initial_noise is not None:
            cum_var_input = initial_noise.float()
        else:
            cum_var_input = torch.zeros_like(target)

        new_bit_indices = []

        for j in range(restart_stage):
            pt, ph, pw = vae_scale_schedule[j]

            # Determine which quantizer head to use
            if getattr(args, 'use_two_stage_lfq', True):
                if ph * pw >= vae.quantizer.detail_scale_min_tokens:
                    is_semantic_scale = False
                    C1 = vae.quantizer.detail_scale_dim
                    lfq = vae.quantizer.lfq_detail
                else:
                    is_semantic_scale = True
                    C1 = vae.quantizer.semantic_scale_dim
                    lfq = vae.quantizer.lfq_semantic
            else:
                is_semantic_scale = False
                lfq = vae.quantizer.lfq_detail
                C1 = summed_code.shape[1]

            # Upsample cum_var_input to target resolution for residual computation
            if cum_var_input.shape[-3:] != target.shape[-3:]:
                cum_interp = F.interpolate(cum_var_input, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up)
            else:
                cum_interp = cum_var_input

            # Compute residual at target resolution
            residual = target - cum_interp

            # Downsample and project residual to quantizer scale
            if getattr(args, 'use_two_stage_lfq', True):
                residual_ds = interpolate_fn(
                    residual, size=(C1, pt, ph, pw),
                    mode=vae.quantizer.z_interplote_down,
                    quantizer=vae.quantizer,
                    is_semantic_scale=is_semantic_scale
                ).contiguous()
            else:
                residual_ds = F.interpolate(residual, size=(pt, ph, pw), mode=vae.quantizer.z_interplote_down).contiguous()

            # Quantize
            quantized, _, bit_indices, _ = lfq(residual_ds)
            new_bit_indices.append(bit_indices)

            # Upsample quantized codes back to target resolution and accumulate
            if getattr(args, 'use_two_stage_lfq', True):
                quantized_up = interpolate_fn(
                    quantized, size=(target.shape[1], *target_pn),
                    mode=vae.quantizer.z_interplote_up,
                    quantizer=vae.quantizer,
                    is_semantic_scale=is_semantic_scale
                ).contiguous()
            else:
                quantized_up = F.interpolate(quantized, size=target_pn, mode=vae.quantizer.z_interplote_up).contiguous()

            if cum_var_input.shape[-3:] != target_pn:
                cum_var_input = F.interpolate(cum_var_input, size=target_pn, mode=vae.quantizer.z_interplote_up)
            cum_var_input = cum_var_input + quantized_up

        new_summed_code = cum_var_input

    return new_bit_indices, new_summed_code.to(summed_code.dtype)
