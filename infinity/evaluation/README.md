# Evaluation notes

`evaluate.py` at the integration root is the single entry point; this package only holds the metric
implementations. The exact same `evaluate.py` and `evaluation/eval_metrics.py` ship in the Switti
integration, so both image models are evaluated with the same command and produce the same
`scores.json` schema.

```bash
python evaluate.py --save_path outputs/infinity/example \
  --metrics LPIPS,Vendi,ImageReward,FID --fid_dataset mjhq-30k
```

Supported metrics:

| `--metrics` name | Reported keys | Notes |
| --- | --- | --- |
| `LPIPS` | `LPIPS_Mean`, `LPIPS_Std` | LPIPS-MPD: mean pairwise LPIPS distance within each prompt group. |
| `Vendi` | `Vendi`, `Vendi_MSS` | Vendi score over SSCD embeddings; requires `DIVERSEVAR_SSCD_PATH`. |
| `ImageReward` | `ImageReward` | Mean ImageReward score over all generated images. |
| `FID` | `FID` | `--fid_dataset ms-coco` (T2IBenchmark) or `--fid_dataset mjhq-30k` (clean-fid). |

Useful flags:

- `--image_exts`: defaults to `auto`, which detects `png`, `jpg` or `jpeg` in the folder.
- `--fid_dataset`: `ms-coco` or `mjhq-30k`, auto-detected from the prompt-file name in the path.
- `--scores_path` and `--label`: write the scores elsewhere, and tag them for later aggregation.
- `--overwrite`: recompute metrics that are already in the scores file (runs resume by default).

Metric models are loaded on demand, so only the packages and checkpoints for the metrics you select
need to be installed. `DIVERSEVAR_SSCD_PATH` must point at the SSCD TorchScript checkpoint for the
Vendi metric.

Video metrics (VBench and VBench-2.0) are not part of this repository; run their official
implementations on the generated videos.
