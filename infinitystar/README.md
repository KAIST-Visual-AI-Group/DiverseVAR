# DiverseVAR for InfinityStar

This directory adapts [InfinityStar](https://github.com/FoundationVision/InfinityStar) for diverse text-to-video generation. DiverseVAR is applied to the keyframe stage, as described in the paper; the code also exposes experimental temporal-stage controls.

## Setup

```bash
conda env create -f environment.yml
conda activate diversevar-infinitystar
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Download the InfinityStar repository files:

```bash
huggingface-cli download FoundationVision/InfinityStar \
  --local-dir pretrained/InfinityStar
```

The expected directory contains `infinitystar_8b_480p_weights/`, `infinitystar_videovae.pth`, and `text_encoder/flan-t5-xl-official/`.

## Inference

Prompts are read from a text file, one prompt per line, exactly as in the Infinity and Switti
integrations. Example lists live in the shared [`../prompts/`](../prompts) directory.

```bash
python tools/infer_video_480p.py \
  --checkpoints_dir pretrained/InfinityStar \
  --prompt_txt ../prompts/example-video-prompt.txt \
  --save_dir outputs/infinitystar --duration 5 --num_seeds 3 \
  --noise_type cads --noise_seed 990730 \
  --noise_scale 1.0 --noise_stage 8 \
  --anneal_func cosine \
  --repeat_num 2 --encode_stage 12 --restart_stage 6
```

This matches the paper's InfinityStar qualitative setup: condition annealing and
scale-travel are applied only to keyframe generation. `example-video-prompt.txt` holds the
motion-aware augmented prompt denoted by † in Figure 7. Use
`--noise_type none --noise_scale 0 --repeat_num 1` for the InfinityStar baseline.
Leave `--noise_stage_video 0 --repeat_num_video 1` unless explicitly studying
temporal-stage variants.

## VBench evaluation

VBench is not included in this repository. Install it separately and generate its inputs with:

```bash
python tools/infer_vbench2_480p.py \
  --worker_id 0 --num_workers 1 \
  --checkpoints_dir pretrained/InfinityStar \
  --vbench_prompt_dir /path/to/VBench-2.0/prompts/prompt \
  --save_dir outputs/vbench2 \
  --dimensions Diversity --max_prompts_per_dim 1 \
  --noise_type cads --noise_seed 990730 \
  --noise_scale 1.0 --noise_stage 8 \
  --anneal_func cosine --repeat_num 2 --encode_stage 12 --restart_stage 6
```

Then follow the official VBench/VBench-2.0 evaluator instructions on `outputs/vbench2`. Keeping VBench external avoids shipping its large third-party dependency tree inside this implementation.

Full video generation was designed for high-memory CUDA hardware; an 8B video model is not
expected to fit on a single 24 GB GPU without additional offloading.

## Training

This release contains inference and evaluation code only. For training and dataset preparation,
use the upstream [InfinityStar](https://github.com/FoundationVision/InfinityStar) repository.

The original InfinityStar license is retained in `LICENSE`.
