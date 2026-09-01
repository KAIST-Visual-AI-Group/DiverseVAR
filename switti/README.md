# DiverseVAR for Switti

This directory adapts [Switti](https://github.com/yandex-research/switti) with condition annealing and scale-travel refinement at inference time. The implementation is integrated into `models/pipeline.py`; schedules and CLI options live in `utils_local/diverse_var_utils.py`.

This release contains inference and evaluation code only. For training and dataset preparation, use the upstream [Switti](https://github.com/yandex-research/switti) repository.

## Setup

```bash
conda env create -f environment.yml
conda activate diversevar-switti
pip install -r requirements.txt
```

`yresearch/Switti-1024` and its VQ-VAE dependencies are downloaded automatically from Hugging Face on first use. If the repository is gated in the future, authenticate interactively with `huggingface-cli login`; never put a token in source code.

## Inference

Prompt lists live in the shared [`../prompts/`](../prompts) directory (one prompt per line):
`example-image-prompt.txt`, `ms-coco-1000-prompts.txt`, `mjhq-mixed-1000-prompts.txt`.

```bash
python inference.py \
  --model_path yresearch/Switti-1024 \
  --prompt_txt ../prompts/example-image-prompt.txt \
  --save_path outputs/switti \
  --imgs_per_prompt 4 --save_indv 1 \
  --noise_type context --noise_scale 1.0 --noise_stage 6 \
  --anneal_func cosine --noise_rescale 1.0 \
  --repeat_num 2 --encode_stage 8 --restart_stage 3
```

For the Switti baseline, set `--noise_type none --noise_scale 0 --repeat_num 1`. The output directory contains images and `metadata.json`.

## Evaluation

`evaluate.py` is the same file in the Switti and Infinity integrations: identical CLI, identical
`evaluation/eval_metrics.py`, identical `scores.json` schema. It reads the generated
`metadata.json`, derives the prompt count and images-per-prompt from it, and writes `scores.json`
next to the images (`metadata.json` is never modified).

```bash
python evaluate.py --save_path <generated-image-directory> \
  --metrics LPIPS,Vendi,ImageReward,FID --fid_dataset mjhq-30k
```

`--image_exts` defaults to `auto` (detects `jpg`/`png`), `--fid_dataset` is auto-detected from the
prompt-file name in the path, and existing scores are reused unless `--overwrite` is passed. The
same evaluator runs in-process after generation with `--eval 1` on `inference.py`
(`--eval_metrics` selects what is computed).

The metrics are LPIPS-MPD, Vendi (SSCD features), ImageReward, and FID. Backends are loaded on
demand and may download their own encoders. Vendi requires `DIVERSEVAR_SSCD_PATH` to point to an
SSCD TorchScript checkpoint. FID uses T2IBenchmark COCO statistics for `--fid_dataset ms-coco` and
clean-fid custom statistics for `--fid_dataset mjhq-30k`; reference statistics are not bundled.
Video results are measured externally with VBench / VBench-2.0.

The original Switti license is retained in `LICENSE`.
