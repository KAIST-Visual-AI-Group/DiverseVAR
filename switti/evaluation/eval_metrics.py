"""Metric implementations shared by the DiverseVAR image integrations.

This module is a library only; `evaluate.py` at the integration root is the
single entry point. Every metric operates on one flat folder shaped like

    <save_path>/
        metadata.json           # {"images": {"PPPPP_II": {"prompt", "seed"}}}
        00000_00.png
        00000_01.png
        ...

Diversity metrics (LPIPS-MPD, Vendi) group images per prompt by sorting the
files lexicographically and slicing them into chunks of `imgs_per_prompt`, so
the zero-padded `PPPPP_II` naming is required. Each metric returns a dict of
score keys for the whole folder.
"""

import json
import os
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from vendi_score import vendi

try:
    from .DAS_scorer import ImageRewardScorer
except ImportError:  # Allow `python evaluation/eval_metrics.py`-style imports.
    from DAS_scorer import ImageRewardScorer


def preprocess_image_lpips(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def preprocess_image_vendi(image_path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def group_images_by_prompt(image_folder, image_exts, prompts, imgs_per_prompt):
    """Return [[image_path, ...], ...], one list of samples per prompt."""
    img_list = sorted(Path(image_folder).glob("*" + image_exts))
    return [
        img_list[p * imgs_per_prompt:(p + 1) * imgs_per_prompt]
        for p in range(prompts)
    ]


@torch.no_grad()
def lpips_mean_eval(image_folder, image_exts='.png', prompts=1, imgs_per_prompt=1):
    """LPIPS-MPD: mean pairwise LPIPS distance inside each prompt group.

    Distances from every prompt are pooled; with a constant `imgs_per_prompt`
    the mean equals the mean of the per-prompt means.
    """
    lpips_model = lpips.LPIPS(net='alex').to("cuda")

    all_distances = []
    for paths in group_images_by_prompt(image_folder, image_exts, prompts, imgs_per_prompt):
        images = [preprocess_image_lpips(p) for p in paths]
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                all_distances.append(lpips_model(images[i].to("cuda"), images[j].to("cuda")).item())

    del lpips_model
    torch.cuda.empty_cache()

    if not all_distances:
        raise RuntimeError(f"No image pairs found in {image_folder} (ext={image_exts})")

    return {
        "LPIPS_Mean": float(np.mean(all_distances)),
        "LPIPS_Std": float(np.std(all_distances)),
    }


@torch.no_grad()
def vendi_eval(image_folder, image_exts='.png', prompts=1, imgs_per_prompt=1):
    """Vendi score over SSCD embeddings, averaged over prompts.

    `Vendi_MSS` is the mean pairwise cosine similarity of the same embeddings.
    """
    checkpoint = os.environ.get("DIVERSEVAR_SSCD_PATH")
    if not checkpoint:
        raise RuntimeError("Set DIVERSEVAR_SSCD_PATH to the SSCD TorchScript checkpoint")
    scorer = torch.jit.load(checkpoint).cuda().eval()

    vendi_scores, mss_scores = [], []
    for paths in group_images_by_prompt(image_folder, image_exts, prompts, imgs_per_prompt):
        if not paths:
            continue
        embeddings = torch.stack([scorer(preprocess_image_vendi(p).to("cuda"))[0] for p in paths])
        normalized = F.normalize(embeddings, dim=1)
        cosine_sim_matrix = (normalized @ normalized.T).cpu().numpy()
        vendi_scores.append(float(vendi.score_K(cosine_sim_matrix)))
        mss_scores.append(float(np.average(cosine_sim_matrix)))

    del scorer
    torch.cuda.empty_cache()

    if not vendi_scores:
        raise RuntimeError(f"No images found in {image_folder} (ext={image_exts})")

    return {
        "Vendi": float(np.mean(vendi_scores)),
        "Vendi_MSS": float(np.mean(mss_scores)),
    }


def images_to_tensor_with_prompt(image_folder, image_exts='.png'):
    """Load every image listed in metadata.json together with its prompt."""
    image_folder = Path(image_folder)
    with open(image_folder / "metadata.json", 'r') as f:
        metadata = json.load(f)

    images, prompts = [], []
    for img_name in sorted(metadata["images"].keys()):
        img_path = image_folder / (img_name + image_exts)
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        images.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1))
        prompts.append(metadata["images"][img_name]["prompt"])

    if not images:
        raise RuntimeError(f"No images from metadata.json found in {image_folder} (ext={image_exts})")
    return images, prompts


@torch.no_grad()
def image_reward_eval(image_folder, image_exts='.png', prompts=1, imgs_per_prompt=1):
    """Mean ImageReward over every generated image."""
    images, prompt_texts = images_to_tensor_with_prompt(image_folder, image_exts)
    scorer = ImageRewardScorer(torch.float32, "cuda")

    scores = [
        scorer(image.unsqueeze(0).to("cuda"), prompt).item()
        for image, prompt in zip(images, prompt_texts)
    ]

    del scorer
    torch.cuda.empty_cache()

    return {"ImageReward": float(np.mean(scores))}


def fid_eval(image_folder, fid_dataset):
    """FID against a benchmark reference set, over the whole folder.

    ms-coco  : precomputed InceptionV3 stats of MS-COCO val2014 30K (T2IBenchmark).
    mjhq-30k : clean-fid custom stats you build once with
               `cleanfid.fid.make_custom_stats("mjhq-30k", <ref-dir>, mode="clean")`.
    """
    if fid_dataset == "ms-coco":
        from T2IBenchmark import calculate_fid
        from T2IBenchmark.datasets import get_coco_fid_stats
        fid_score, _ = calculate_fid(str(image_folder), get_coco_fid_stats())
        return float(fid_score)

    if fid_dataset == "mjhq-30k":
        from cleanfid import fid
        return float(fid.compute_fid(
            str(image_folder),
            dataset_name="mjhq-30k",
            mode="clean",
            dataset_split="custom",
        ))

    raise ValueError(f"Unknown FID reference dataset: {fid_dataset}")
