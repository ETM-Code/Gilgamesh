#!/usr/bin/env python3
"""
Create 6x6 downsampled MNIST dataset in IDX format.

The IDX format is the original MNIST binary format:
- train-images-idx3-ubyte: training images
- train-labels-idx1-ubyte: training labels
- t10k-images-idx3-ubyte: test images
- t10k-labels-idx1-ubyte: test labels

This script downloads the original 28x28 MNIST and creates 6x6 versions.
"""

import struct
import gzip
import numpy as np
from pathlib import Path

try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    USE_TORCHVISION = False
    print("torchvision not found, using mnist package")
    import mnist


def downsample_images(images: np.ndarray, target_size: int = 6) -> np.ndarray:
    """Downsample images from 28x28 to target_size x target_size by averaging."""
    n_images = images.shape[0]
    original_size = 28
    scale = original_size // target_size

    # Reshape for block averaging: (n, 6, 4, 6, 4) -> mean over axes 2,4
    # This is equivalent to 4x4 block averaging for 28->6 (with 4 leftover pixels ignored)

    result = np.zeros((n_images, target_size, target_size), dtype=np.float32)

    for i in range(n_images):
        for ty in range(target_size):
            for tx in range(target_size):
                # Average the scale x scale block
                block = images[i, ty*scale:(ty+1)*scale, tx*scale:(tx+1)*scale]
                result[i, ty, tx] = block.mean()

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{n_images} images")

    return result


def write_idx_images(filepath: Path, images: np.ndarray):
    """Write images to IDX3 format (gzipped)."""
    n_images, rows, cols = images.shape

    # Convert to uint8 (0-255)
    images_uint8 = (images * 255).astype(np.uint8)

    with gzip.open(filepath, 'wb') as f:
        # Magic number for idx3-ubyte: 2051 (0x00000803)
        f.write(struct.pack('>I', 2051))
        f.write(struct.pack('>I', n_images))
        f.write(struct.pack('>I', rows))
        f.write(struct.pack('>I', cols))
        f.write(images_uint8.tobytes())

    print(f"  Written {filepath} ({n_images} images, {rows}x{cols})")


def write_idx_labels(filepath: Path, labels: np.ndarray):
    """Write labels to IDX1 format (gzipped)."""
    n_labels = labels.shape[0]
    labels_uint8 = labels.astype(np.uint8)

    with gzip.open(filepath, 'wb') as f:
        # Magic number for idx1-ubyte: 2049 (0x00000801)
        f.write(struct.pack('>I', 2049))
        f.write(struct.pack('>I', n_labels))
        f.write(labels_uint8.tobytes())

    print(f"  Written {filepath} ({n_labels} labels)")


def load_mnist_torchvision(data_dir: Path):
    """Load MNIST using torchvision."""
    train_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=False, download=True)

    train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()

    test_images = test_dataset.data.numpy().astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy()

    return train_images, train_labels, test_images, test_labels


def load_mnist_package():
    """Load MNIST using mnist package."""
    train_images = mnist.train_images().astype(np.float32) / 255.0
    train_labels = mnist.train_labels()
    test_images = mnist.test_images().astype(np.float32) / 255.0
    test_labels = mnist.test_labels()

    return train_images, train_labels, test_images, test_labels


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Output directory for 6x6 dataset
    output_dir = project_dir / "data_6x6"
    output_dir.mkdir(exist_ok=True)

    # Original data directory (for downloading)
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("Loading original 28x28 MNIST...")
    if USE_TORCHVISION:
        train_images, train_labels, test_images, test_labels = load_mnist_torchvision(data_dir)
    else:
        train_images, train_labels, test_images, test_labels = load_mnist_package()

    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    print("\nDownsampling training images to 6x6...")
    train_images_6x6 = downsample_images(train_images, target_size=6)

    print("\nDownsampling test images to 6x6...")
    test_images_6x6 = downsample_images(test_images, target_size=6)

    print(f"\nDownsampled shapes: Train {train_images_6x6.shape}, Test {test_images_6x6.shape}")

    print("\nWriting IDX files...")
    write_idx_images(output_dir / "train-images-idx3-ubyte.gz", train_images_6x6)
    write_idx_labels(output_dir / "train-labels-idx1-ubyte.gz", train_labels)
    write_idx_images(output_dir / "t10k-images-idx3-ubyte.gz", test_images_6x6)
    write_idx_labels(output_dir / "t10k-labels-idx1-ubyte.gz", test_labels)

    # Also write uncompressed versions (some loaders prefer these)
    print("\nWriting uncompressed IDX files...")

    def write_idx_images_raw(filepath: Path, images: np.ndarray):
        n_images, rows, cols = images.shape
        images_uint8 = (images * 255).astype(np.uint8)
        with open(filepath, 'wb') as f:
            f.write(struct.pack('>I', 2051))
            f.write(struct.pack('>I', n_images))
            f.write(struct.pack('>I', rows))
            f.write(struct.pack('>I', cols))
            f.write(images_uint8.tobytes())
        print(f"  Written {filepath}")

    def write_idx_labels_raw(filepath: Path, labels: np.ndarray):
        n_labels = labels.shape[0]
        labels_uint8 = labels.astype(np.uint8)
        with open(filepath, 'wb') as f:
            f.write(struct.pack('>I', 2049))
            f.write(struct.pack('>I', n_labels))
            f.write(labels_uint8.tobytes())
        print(f"  Written {filepath}")

    write_idx_images_raw(output_dir / "train-images-idx3-ubyte", train_images_6x6)
    write_idx_labels_raw(output_dir / "train-labels-idx1-ubyte", train_labels)
    write_idx_images_raw(output_dir / "t10k-images-idx3-ubyte", test_images_6x6)
    write_idx_labels_raw(output_dir / "t10k-labels-idx1-ubyte", test_labels)

    # Also save as numpy arrays for convenience
    print("\nSaving numpy arrays...")
    np.save(output_dir / "train_images.npy", train_images_6x6)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_images.npy", test_images_6x6)
    np.save(output_dir / "test_labels.npy", test_labels)
    print(f"  Saved .npy files to {output_dir}")

    print(f"\n6x6 MNIST dataset created in {output_dir}")
    print("\nFiles:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
