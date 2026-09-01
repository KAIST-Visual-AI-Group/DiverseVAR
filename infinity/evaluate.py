"""Evaluation entry point shared by the DiverseVAR image integrations.

Both Infinity and Switti write the same output layout, so both are evaluated
with the same command:

    <save_path>/
        metadata.json           # {"images": {"PPPPP_II": {"prompt", "seed"}}}
        00000_00.png
        00000_01.png
        ...

Prompt count and images-per-prompt are derived from metadata.json. Results go
to a SEPARATE scores file (default `<save_path>/scores.json`); metadata.json is
never modified.

Example:
    python evaluate.py --save_path outputs/example \
        --metrics LPIPS,Vendi,ImageReward,FID --fid_dataset mjhq-30k
"""
# --- Workaround: openai/CLIP imports `from pkg_resources import packaging`,
# which breaks on setuptools>=70 because `pkg_resources.packaging` was removed.
# Inject the standalone `packaging` module under `pkg_resources` before anything
# (eval_metrics imports DAS_scorer, which imports ImageReward and clip) tries to import it.
try:
    import pkg_resources as _pr
    if not hasattr(_pr, "packaging"):
        import packaging as _packaging
        _pr.packaging = _packaging
except Exception:
    pass

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_METRICS = ["LPIPS", "Vendi", "ImageReward", "FID"]
FID_DATASETS = ["ms-coco", "mjhq-30k"]
KNOWN_IMAGE_EXTS = ["png", "jpg", "jpeg"]

SCORE_KEYS = {
    "LPIPS": ("LPIPS_Mean", "LPIPS_Std"),
    "Vendi": ("Vendi", "Vendi_MSS"),
    "ImageReward": ("ImageReward",),
    "FID": ("FID",),
}


def parse_args():
    p = argparse.ArgumentParser(description="Run evaluation metrics on a generated-image folder.")
    p.add_argument('--save_path', required=True, type=str,
                   help='Folder containing images and metadata.json.')
    p.add_argument('--metrics', default='LPIPS,Vendi,ImageReward,FID', type=str,
                   help=f'Comma-separated metrics. Choices: {VALID_METRICS}')
    p.add_argument('--image_exts', default='auto', type=str,
                   help='Image extension without dot ("png", "jpg"); "auto" detects it.')
    p.add_argument('--fid_dataset', default=None, choices=FID_DATASETS,
                   help='Reference dataset for FID. If not given, auto-detected from save_path.')
    p.add_argument('--metadata_path', default=None, type=str,
                   help='Override metadata.json path. Defaults to <save_path>/metadata.json.')
    p.add_argument('--scores_path', default=None, type=str,
                   help='Where to write the score JSON. Defaults to <save_path>/scores.json.')
    p.add_argument('--label', default=None, type=str,
                   help='Optional label stored inside scores.json for downstream aggregation.')
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute and overwrite scores already present in the scores file.')
    return p.parse_args()


def load_metadata(save_path: Path, metadata_path=None):
    md_path = Path(metadata_path) if metadata_path else save_path / "metadata.json"
    if not md_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {md_path}")
    with md_path.open("r") as f:
        metadata = json.load(f)
    if "images" not in metadata or not metadata["images"]:
        raise ValueError(f"metadata['images'] missing or empty in {md_path}")
    return metadata


def load_scores(scores_path: Path) -> dict:
    """Load an existing scores file (for resume), or return a fresh skeleton."""
    if scores_path.exists():
        with scores_path.open("r") as f:
            return json.load(f)
    return {"scores": {}}


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def derive_prompt_layout(metadata: dict):
    """From keys like 'PPPPP_II', return (prompt_num, imgs_per_prompt).

    Raises if imgs-per-prompt is inconsistent across prompts, because the
    diversity metrics slice the sorted file list into equal chunks.
    """
    prefixes = Counter(k.rsplit("_", 1)[0] for k in metadata["images"])
    counts = set(prefixes.values())
    if len(counts) != 1:
        breakdown = sorted(prefixes.items())[:5]
        raise ValueError(
            f"Inconsistent imgs-per-prompt across {len(prefixes)} prompts: "
            f"counts={sorted(counts)}, first prefixes={breakdown}"
        )
    return len(prefixes), counts.pop()


def detect_image_ext(save_path: Path) -> str:
    for ext in KNOWN_IMAGE_EXTS:
        if next(save_path.glob(f"*.{ext}"), None) is not None:
            return ext
    raise FileNotFoundError(
        f"No {'/'.join(KNOWN_IMAGE_EXTS)} images in {save_path}; pass --image_exts explicitly."
    )


def detect_fid_dataset(save_path: Path):
    """Infer the FID reference set from the prompt-file name inside save_path."""
    s = str(save_path).lower()
    if "mjhq" in s:
        return "mjhq-30k"
    if "ms-coco" in s or "mscoco" in s or "coco" in s:
        return "ms-coco"
    return None


def evaluate_folder(save_path, metrics, fid_dataset=None, image_exts='auto', overwrite=False,
                    scores_path=None, metadata_path=None, label=None):
    """Score one generation folder and return its scores object."""
    save_path = Path(save_path)
    if not save_path.is_dir():
        raise FileNotFoundError(f"save_path is not a directory: {save_path}")

    unknown = [m for m in metrics if m not in VALID_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid: {VALID_METRICS}")

    metadata = load_metadata(save_path, metadata_path)
    prompt_num, imgs_per_prompt = derive_prompt_layout(metadata)
    ext = detect_image_ext(save_path) if image_exts == 'auto' else image_exts.lstrip('.')

    scores_path = Path(scores_path) if scores_path else save_path / "scores.json"
    scores_obj = load_scores(scores_path)
    scores_obj.setdefault("scores", {})
    scores_obj["save_path"] = str(save_path)
    scores_obj["prompt_num"] = prompt_num
    scores_obj["imgs_per_prompt"] = imgs_per_prompt
    scores_obj["image_exts"] = ext
    if label is not None:
        scores_obj["label"] = label

    print(f"[Eval] save_path={save_path}")
    print(f"[Eval] prompts={prompt_num}, imgs_per_prompt={imgs_per_prompt}, "
          f"total={prompt_num * imgs_per_prompt}, ext=.{ext}")

    # Lazy import so a run that only needs FID does not load the other backends.
    from evaluation.eval_metrics import (
        fid_eval, image_reward_eval, lpips_mean_eval, vendi_eval,
    )

    metric_fns = {
        "LPIPS": lpips_mean_eval,
        "Vendi": vendi_eval,
        "ImageReward": image_reward_eval,
    }
    metric_kwargs = dict(
        image_exts=f'.{ext}',
        prompts=prompt_num,
        imgs_per_prompt=imgs_per_prompt,
    )

    for metric in metrics:
        if not overwrite and all(k in scores_obj["scores"] for k in SCORE_KEYS[metric]):
            existing = {k: scores_obj["scores"][k] for k in SCORE_KEYS[metric]}
            print(f"[Eval] {metric} already present, skipping. Use --overwrite to recompute. {existing}")
            continue

        print(f"[Eval] Running {metric}...", flush=True)

        if metric == "FID":
            dataset = fid_dataset or detect_fid_dataset(save_path)
            if dataset is None:
                print(f"[Eval][WARN] Could not auto-detect the FID reference set from the path; "
                      f"pass --fid_dataset {{{'/'.join(FID_DATASETS)}}}. Skipping FID.")
                continue
            print(f"[Eval]   FID reference dataset: {dataset}")
            try:
                scores_obj["scores"]["FID"] = fid_eval(save_path, dataset)
            except Exception as e:
                print(f"[Eval][WARN] FID failed for {save_path}: {e}", flush=True)
                scores_obj["scores"]["FID"] = -1
        else:
            scores_obj["scores"].update(metric_fns[metric](save_path, **metric_kwargs))

        # Save after each metric so partial progress is never lost.
        save_json(scores_path, scores_obj)
        print(f"[Eval] {metric} done. scores={scores_obj['scores']}", flush=True)

    save_json(scores_path, scores_obj)
    print(f"[Eval] Done -> {scores_path}", flush=True)
    return scores_obj


def evaluate(args):
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    return evaluate_folder(
        args.save_path, metrics,
        fid_dataset=args.fid_dataset, image_exts=args.image_exts,
        overwrite=args.overwrite, scores_path=args.scores_path,
        metadata_path=args.metadata_path, label=args.label,
    )


if __name__ == "__main__":
    evaluate(parse_args())
