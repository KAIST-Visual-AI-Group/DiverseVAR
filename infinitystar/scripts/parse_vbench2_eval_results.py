#!/usr/bin/env python3
"""
Parse VBench-2.0 evaluation results from a directory (e.g. vbench2_evaluation_baseline).
Outputs 5-category scores, total score, and flat mean over 19 dimensions.
"""
import os
import sys
import json
import glob
import argparse
import math

# VBench-2.0 category definitions (from VBench-2.0/scripts/constant.py)
TASK_INFO = [
    "Camera Motion", "Complex Landscape", "Complex Plot", "Composition", "Diversity",
    "Dynamic Attribute", "Dynamic Spatial Relationship", "Human Anatomy", "Human Clothes",
    "Human Identity", "Human Interaction", "Instance Preservation", "Material", "Mechanics",
    "Motion Order Understanding", "Motion Rationality", "Multi-View Consistency", "Thermotics",
]

CREATIVITY_LIST = ["Composition", "Diversity"]
COMMONSENSE_LIST = ["Instance Preservation", "Motion Rationality"]
CONTROLLABILITY_LIST = [
    "Camera Motion", "Complex Landscape", "Complex Plot", "Dynamic Attribute",
    "Dynamic Spatial Relationship", "Human Interaction", "Motion Order Understanding",
]
HUMAN_FIDELITY_LIST = ["Human Anatomy", "Human Clothes", "Human Identity"]
PHYSICS_LIST = ["Material", "Mechanics", "Multi-View Consistency", "Thermotics"]


def parse_eval_dir(eval_root, allow_partial=False):
    """Read available dimension scores; never substitute missing results with zero."""
    upload_data = {}
    missing = []
    invalid = []
    # Folder names use underscore: Camera_Motion, Human_Anatomy, etc.
    # TASK_INFO uses space: "Camera Motion", "Human Anatomy"
    for task in TASK_INFO:
        folder_name = task.replace(" ", "_")
        subdir = os.path.join(eval_root, folder_name)
        if not os.path.isdir(subdir):
            missing.append(f"{task}: missing directory {subdir}")
            continue
        pattern = os.path.join(subdir, "*_eval_results.json")
        files = glob.glob(pattern)
        if not files:
            missing.append(f"{task}: missing result file in {subdir}")
            continue
        try:
            with open(files[0], "r") as f:
                cur_json = json.load(f)
            if not isinstance(cur_json, dict):
                raise ValueError("expected a JSON object")
            # Preserve underscore keys and the space-separated spelling accepted by
            # the original fallback, without misattributing unrelated dimensions.
            key = folder_name if folder_name in cur_json else task
            if key not in cur_json:
                raise ValueError(f"missing JSON key {folder_name!r} or {task!r}")
            val = cur_json[key]
            if isinstance(val, (list, tuple)) and not val:
                raise ValueError("empty score sequence")
            score = val[0] if isinstance(val, (list, tuple)) else val
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("score must be finite")
            upload_data[task] = score
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            invalid.append(f"{task}: {files[0]}: {exc}")
    if invalid:
        raise ValueError("Invalid evaluation results:\n" + "\n".join(invalid + missing))
    if missing:
        message = "Incomplete evaluation results:\n" + "\n".join(missing)
        if not allow_partial:
            raise ValueError(message)
        print(message, file=sys.stderr)
    return upload_data


def main():
    parser = argparse.ArgumentParser(description="Parse VBench-2.0 evaluation results")
    parser.add_argument("eval_dir", type=str, help="Path to evaluation output directory")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Report available dimensions and coverage; withhold aggregates if incomplete")
    args = parser.parse_args()

    eval_root = os.path.abspath(args.eval_dir)
    if not os.path.isdir(eval_root):
        print(f"Not a directory: {eval_root}", file=sys.stderr)
        sys.exit(1)

    try:
        upload_data = parse_eval_dir(eval_root, allow_partial=args.allow_partial)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if len(upload_data) != len(TASK_INFO):
        print(f"Partial evaluation: {len(upload_data)}/{len(TASK_INFO)} dimensions available. "
              "Complete aggregate scores withheld.")
        for task in TASK_INFO:
            if task in upload_data:
                print(f"  {task}: {upload_data[task]:.4f}")
        return

    def cat_mean(keys):
        return sum(upload_data.get(k, 0) for k in keys) / len(keys)

    creativity_score = cat_mean(CREATIVITY_LIST)
    commonsense_score = cat_mean(COMMONSENSE_LIST)
    controllability_score = cat_mean(CONTROLLABILITY_LIST)
    human_fidelity_score = cat_mean(HUMAN_FIDELITY_LIST)
    physics_score = cat_mean(PHYSICS_LIST)
    total_score = (
        creativity_score + commonsense_score + controllability_score
        + human_fidelity_score + physics_score
    ) / 5
    flat_mean = sum(upload_data.get(k, 0) for k in TASK_INFO) / len(TASK_INFO)

    print("Per-dimension scores (raw):")
    for k in TASK_INFO:
        print(f"  {k}: {upload_data.get(k, 0):.4f}")
    print()
    print("+---------------------|------------------+")
    print(f"|     creativity score|{creativity_score:.4f}             |")
    print(f"|    commonsense score|{commonsense_score:.4f}             |")
    print(f"|controllability score|{controllability_score:.4f}             |")
    print(f"| human fidelity score|{human_fidelity_score:.4f}             |")
    print(f"|        physics score|{physics_score:.4f}             |")
    print(f"|          total score|{total_score:.4f}             |")
    print(f"|      flat mean (19d)|{flat_mean:.4f}             |")
    print("+---------------------|------------------+")


if __name__ == "__main__":
    main()
