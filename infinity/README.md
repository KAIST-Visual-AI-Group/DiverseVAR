# DiverseVAR for Infinity

This directory adapts [Infinity](https://github.com/FoundationVision/Infinity) with the training-free DiverseVAR inference procedure. The main additions are condition annealing and scale-travel refinement in `infinity/models/infinity.py`, with reusable schedules in `infinity/utils/diverse_var_utils.py`.

This release contains inference and evaluation code only. For training and dataset preparation, use the upstream [Infinity](https://github.com/FoundationVision/Infinity) repository.

## Setup

```bash
conda env create -f environment.yml
conda activate diversevar-infinity
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Download the Infinity weights from [FoundationVision/Infinity](https://huggingface.co/FoundationVision/Infinity). For the 2B model, place `infinity_2b_reg.pth` and `infinity_vae_d32reg.pth` under `weights/`. The T5 encoder defaults to `google/flan-t5-xl` and is downloaded by Transformers on first use.

## Inference

Prompt lists live in the shared [`../prompts/`](../prompts) directory (one prompt per line):
`example-image-prompt.txt`, `ms-coco-1000-prompts.txt`, `mjhq-mixed-1000-prompts.txt`.
A compact DiverseVAR example is:

```bash
python inference.py \
  --network_type infinity_2b \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --prompt_txt ../prompts/example-image-prompt.txt \
  --save_path outputs/infinity \
  --imgs_per_prompt 4 --save_indv 1 \
  --noise_type cads --noise_scale 1.0 --noise_stage 4 \
  --anneal_func cosine --noise_rescale 1.0 \
  --repeat_num 2 --encode_stage 8 --restart_stage 2
```

For the Infinity baseline, set `--noise_type none --noise_scale 0 --repeat_num 1`. Outputs include images and `metadata.json`; the nested output directory records all sampling settings.

## Evaluation

`evaluate.py` is the same file in the Infinity and Switti integrations: identical CLI, identical
`evaluation/eval_metrics.py`, identical `scores.json` schema. It reads the generated
`metadata.json`, derives the prompt count and images-per-prompt from it, and writes
`scores.json` next to the images (`metadata.json` is never modified).

```bash
python evaluate.py --save_path <generated-image-directory> \
  --metrics LPIPS,Vendi,ImageReward,FID --fid_dataset mjhq-30k
```

`--image_exts` defaults to `auto` (detects `png`/`jpg`), `--fid_dataset` is auto-detected from the
prompt-file name in the path, and existing scores are reused unless `--overwrite` is passed. The
same evaluator runs in-process after generation with `--eval 1` on `inference.py`
(`--eval_metrics` selects what is computed).

The metrics are LPIPS-MPD, Vendi (SSCD), ImageReward, and FID. Backends are loaded on demand, so
only the requested metric packages need to be installed. Vendi requires `DIVERSEVAR_SSCD_PATH` to
point to an SSCD TorchScript checkpoint. See `evaluation/README.md` for per-metric details.

The original Infinity license is retained in `LICENSE`.
