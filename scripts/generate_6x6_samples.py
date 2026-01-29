#!/usr/bin/env python3
"""Generate 6x6 downsampled MNIST test images for gilgamesh testing."""

import os
import json
import numpy as np
from pathlib import Path

# Try to use torchvision, fall back to mnist package
try:
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    USE_TORCHVISION = True
except ImportError:
    import mnist
    USE_TORCHVISION = False
    from PIL import Image


def downsample_image(img_array: np.ndarray, target_size: int = 6) -> np.ndarray:
    """Downsample a 28x28 image to target_size x target_size by averaging."""
    original_size = 28
    scale = original_size // target_size

    result = np.zeros((target_size, target_size), dtype=np.float32)

    for ty in range(target_size):
        for tx in range(target_size):
            total = 0.0
            count = 0
            for dy in range(scale):
                for dx in range(scale):
                    sy = ty * scale + dy
                    sx = tx * scale + dx
                    if sy < original_size and sx < original_size:
                        total += img_array[sy, sx]
                        count += 1
            result[ty, tx] = total / count

    return result


def normalize_mnist(img: np.ndarray) -> np.ndarray:
    """Apply MNIST normalization: (x - 0.1307) / 0.3081"""
    return (img - 0.1307) / 0.3081


def main():
    # Output directories
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "test_images_6x6"
    output_dir.mkdir(exist_ok=True)

    # Also create a raw pixels directory
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # PNG directory
    png_dir = output_dir / "png"
    png_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load MNIST
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    if USE_TORCHVISION:
        print("Using torchvision to load MNIST...")
        dataset = torchvision.datasets.MNIST(
            root=str(data_dir),
            train=False,  # Use test set
            download=True
        )

        def get_image_and_label(idx):
            img, label = dataset[idx]
            return np.array(img, dtype=np.float32) / 255.0, label
    else:
        print("Using mnist package to load MNIST...")
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        def get_image_and_label(idx):
            return test_images[idx].astype(np.float32) / 255.0, int(test_labels[idx])

    # Generate samples - one of each digit (0-9) plus a few extras
    samples_per_digit = 2
    samples = []
    digit_counts = {d: 0 for d in range(10)}

    idx = 0
    while any(c < samples_per_digit for c in digit_counts.values()):
        img_28x28, label = get_image_and_label(idx)

        if digit_counts[label] < samples_per_digit:
            # Downsample to 6x6
            img_6x6 = downsample_image(img_28x28, target_size=6)

            # Save normalized version (what the network sees)
            img_normalized = normalize_mnist(img_6x6)

            sample_id = f"digit_{label}_{digit_counts[label]}"

            # Save as JSON (normalized, flattened - ready for network input)
            json_path = raw_dir / f"{sample_id}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "label": label,
                    "pixels": img_normalized.flatten().tolist(),
                    "shape": [6, 6],
                    "normalized": True,
                    "normalization": {"mean": 0.1307, "std": 0.3081}
                }, f, indent=2)

            # Save as PNG (visual, 0-255 range, scaled up for visibility)
            img_visual = (img_6x6 * 255).astype(np.uint8)
            # Scale up 10x for visibility
            img_large = np.repeat(np.repeat(img_visual, 10, axis=0), 10, axis=1)
            png_path = png_dir / f"{sample_id}.png"
            Image.fromarray(img_large, mode='L').save(png_path)

            samples.append({
                "id": sample_id,
                "label": label,
                "json_path": str(json_path.relative_to(output_dir)),
                "png_path": str(png_path.relative_to(output_dir))
            })

            digit_counts[label] += 1
            print(f"  Generated {sample_id}")

        idx += 1

    # Save manifest
    manifest = {
        "description": "6x6 downsampled MNIST test images for gilgamesh",
        "image_size": 6,
        "total_pixels": 36,
        "normalization": {
            "mean": 0.1307,
            "std": 0.3081,
            "formula": "(pixel - mean) / std"
        },
        "samples": samples
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(samples)} test images")
    print(f"Manifest: {manifest_path}")
    print(f"Raw JSON files: {raw_dir}")
    print(f"PNG previews: {png_dir}")


if __name__ == "__main__":
    main()
