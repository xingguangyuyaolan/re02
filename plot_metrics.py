#!/usr/bin/env python3
"""Generate PNG plots from a training run's metrics.jsonl."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.scripts.attention_qmix import _read_jsonl, render_training_plots


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from QMIX training metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Training run directory, e.g. artifacts/qmix/stage1_open_seed7_...")
    group.add_argument("--metrics", type=str, help="Direct path to metrics.jsonl")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.run_dir:
        run_dir = os.path.normpath(args.run_dir)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
    else:
        metrics_path = os.path.normpath(args.metrics)
        run_dir = os.path.dirname(metrics_path)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    episode_metrics = _read_jsonl(metrics_path)
    if not episode_metrics:
        raise RuntimeError(f"No valid metrics records found in: {metrics_path}")

    generated = render_training_plots(run_dir, episode_metrics)
    if not generated:
        raise RuntimeError("No plots were generated from the available metrics.")

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()