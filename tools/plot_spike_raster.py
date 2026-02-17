#!/usr/bin/env python3
"""
Render Gilgamesh spike raster plots from spike_raster_dump JSON.

Example:
  python3 tools/plot_spike_raster.py \
    --input ./spike_raster_data.json \
    --background "/Users/eoghancollins/Tarskii/Screenshot 2026-02-10 at 19.25.10.png" \
    --output ./spike_raster_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def denormalize_pixels(p: np.ndarray) -> np.ndarray:
    # Inverse of MNIST normalization in src/data.rs:
    # normalized = (x - 0.1307) / 0.3081
    return np.clip(p * 0.3081 + 0.1307, 0.0, 1.0)


def plot_sample(
    ax_img: plt.Axes,
    ax_hidden: plt.Axes,
    ax_output: plt.Axes,
    sample: dict[str, Any],
    image_w: int,
    image_h: int,
    num_steps: int,
    hidden_size: int,
    output_size: int,
) -> None:
    pixels = np.array(sample["input_pixels"], dtype=np.float32)
    img = denormalize_pixels(pixels).reshape(image_h, image_w)
    ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        f"s{sample['sample_index']}  y={sample['label']}  p={sample['prediction']}",
        fontsize=10,
        color="#EAF6FF",
    )

    hidden_events = np.array(sample["hidden_events"], dtype=np.int32)
    if hidden_events.size > 0:
        ax_hidden.scatter(
            hidden_events[:, 0],
            hidden_events[:, 1],
            marker="|",
            s=120,
            linewidths=1.1,
            color="#47F5C3",
        )
    ax_hidden.set_xlim(-0.5, num_steps - 0.5)
    ax_hidden.set_ylim(-0.5, hidden_size - 0.5)
    ax_hidden.set_ylabel("hidden", color="#BFE8FF", fontsize=9)
    ax_hidden.grid(color="#A0E0FF", alpha=0.15, linewidth=0.5)
    ax_hidden.tick_params(colors="#C8EBFF", labelsize=8)

    output_events = np.array(sample["output_events"], dtype=np.int32)
    if output_events.size > 0:
        ax_output.scatter(
            output_events[:, 0],
            output_events[:, 1],
            marker="|",
            s=140,
            linewidths=1.4,
            color="#FFD25E",
        )
    ax_output.set_xlim(-0.5, num_steps - 0.5)
    ax_output.set_ylim(-0.5, output_size - 0.5)
    ax_output.set_xlabel("timestep", color="#BFE8FF", fontsize=9)
    ax_output.set_ylabel("output", color="#BFE8FF", fontsize=9)
    ax_output.grid(color="#A0E0FF", alpha=0.15, linewidth=0.5)
    ax_output.tick_params(colors="#C8EBFF", labelsize=8)

    spike_counts = sample.get("output_spike_count", [])
    if spike_counts:
        top_cls = int(np.argmax(np.array(spike_counts)))
        total = int(round(float(np.sum(spike_counts))))
        ax_output.set_title(
            f"top={top_cls} total_out_spikes={total}",
            fontsize=9,
            color="#FFDFA2",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to spike_raster_dump JSON")
    parser.add_argument("--output", default="./spike_raster_plot.png", help="Output PNG path")
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background image path",
    )
    parser.add_argument(
        "--bg-alpha",
        type=float,
        default=0.18,
        help="Background alpha (0-1)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    if not samples:
        raise RuntimeError("No samples in input JSON")

    num_steps = int(data["num_steps"])
    hidden_size = int(data["hidden_size"])
    output_size = int(data["output_size"])
    image_w = int(data["image_width"])
    image_h = int(data["image_height"])

    n = len(samples)
    fig = plt.figure(figsize=(13.5, 2.8 * n), constrained_layout=True)

    if args.background:
        bg_path = Path(args.background)
        if bg_path.exists():
            bg = plt.imread(bg_path)
            bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-1)
            bg_ax.imshow(bg, aspect="auto", alpha=float(np.clip(args.bg_alpha, 0.0, 1.0)))
            bg_ax.axis("off")

    gs = fig.add_gridspec(
        nrows=n,
        ncols=3,
        width_ratios=[1.0, 4.5, 4.5],
        hspace=0.28,
        wspace=0.15,
    )

    for i, sample in enumerate(samples):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_hidden = fig.add_subplot(gs[i, 1])
        ax_output = fig.add_subplot(gs[i, 2], sharex=ax_hidden)

        for ax in (ax_img, ax_hidden, ax_output):
            ax.set_facecolor((0.03, 0.05, 0.12, 0.72))
            for spine in ax.spines.values():
                spine.set_color("#78CFFF")
                spine.set_alpha(0.35)

        plot_sample(
            ax_img=ax_img,
            ax_hidden=ax_hidden,
            ax_output=ax_output,
            sample=sample,
            image_w=image_w,
            image_h=image_h,
            num_steps=num_steps,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        if i < n - 1:
            ax_hidden.tick_params(labelbottom=False)
            ax_output.tick_params(labelbottom=False)

    fig.suptitle(
        "Gilgamesh Spike Raster (Correctly Classified Digits)",
        fontsize=15,
        color="#EAF6FF",
    )
    fig.patch.set_facecolor("#0B1020")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote raster plot to {out_path}")


if __name__ == "__main__":
    main()
