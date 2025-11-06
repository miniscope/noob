#!/usr/bin/env python
"""
Generate synthetic test data for examples.

This script creates:
- Synthetic satellite images (with/without fire patterns)
- Synthetic datasets for ML training
- Sample configurations
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_satellite_images(output_dir: Path, n_images: int = 100):
    """
    Generate synthetic satellite images for fire detection demo.

    Creates realistic-looking satellite imagery with:
    - Forest/vegetation textures
    - Fire patterns (20% of images have fire)
    - Smoke effects
    - Terrain variation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🛰️  Generating {n_images} synthetic satellite images...")

    np.random.seed(42)

    for i in range(n_images):
        # Create base image (512x512)
        img = Image.new('RGB', (512, 512))
        pixels = img.load()

        # Generate terrain
        for y in range(512):
            for x in range(512):
                # Forest green with variation
                base_green = 40 + int(30 * np.sin(x / 20) * np.cos(y / 20))
                r = base_green + np.random.randint(-10, 10)
                g = base_green + 80 + np.random.randint(-10, 10)
                b = base_green + np.random.randint(-10, 10)

                pixels[x, y] = (
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b))
                )

        # 20% chance of fire
        has_fire = np.random.random() < 0.2

        if has_fire:
            # Add fire hotspot
            draw = ImageDraw.Draw(img)

            # Random fire location
            fire_x = np.random.randint(100, 412)
            fire_y = np.random.randint(100, 412)
            fire_radius = np.random.randint(20, 60)

            # Fire core (bright orange/red)
            for r in range(fire_radius, 0, -5):
                intensity = r / fire_radius
                color = (
                    255,
                    int(140 * intensity),
                    0
                )
                draw.ellipse(
                    [fire_x - r, fire_y - r, fire_x + r, fire_y + r],
                    fill=color
                )

            # Smoke (gray)
            smoke_x = fire_x + np.random.randint(-30, 30)
            smoke_y = fire_y - np.random.randint(40, 80)
            smoke_radius = fire_radius + 20

            draw.ellipse(
                [smoke_x - smoke_radius, smoke_y - smoke_radius,
                 smoke_x + smoke_radius, smoke_y + smoke_radius],
                fill=(100, 100, 100, 128)
            )

        # Save image
        filename = f"satellite_{i:04d}_{'fire' if has_fire else 'normal'}.jpg"
        img.save(output_dir / filename, quality=85)

        if (i + 1) % 20 == 0:
            print(f"   Generated {i + 1}/{n_images} images...")

    print(f"   ✓ Saved {n_images} images to {output_dir}")

    # Create manifest
    manifest = {
        "total_images": n_images,
        "with_fire": sum(1 for f in output_dir.glob("*_fire.jpg")),
        "image_size": [512, 512],
        "generated_with": "generate_test_data.py"
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"   ✓ Created manifest: {manifest}")


def generate_ml_datasets(output_dir: Path, n_workers: int = 10):
    """
    Generate synthetic datasets for federated learning demo.

    Creates:
    - Multiple worker datasets (simulating distributed data)
    - Sentiment analysis data (text features + labels)
    - Realistic data distribution
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Generating {n_workers} worker datasets for ML training...")

    np.random.seed(42)

    # Generate shared vocabulary (features)
    n_features = 100

    for worker_id in range(n_workers):
        # Each worker has 1000 samples
        n_samples = 1000

        # Generate synthetic sentiment data
        # Positive sentiment: higher values in first 50 features
        # Negative sentiment: higher values in last 50 features
        positive_samples = n_samples // 2
        negative_samples = n_samples - positive_samples

        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        # Positive samples
        for i in range(positive_samples):
            data[i, :50] = np.random.randn(50) * 2 + 3  # High values
            data[i, 50:] = np.random.randn(50) * 2 - 1  # Low values
            labels[i] = 1

        # Negative samples
        for i in range(positive_samples, n_samples):
            data[i, :50] = np.random.randn(50) * 2 - 1  # Low values
            data[i, 50:] = np.random.randn(50) * 2 + 3  # High values
            labels[i] = 0

        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        # Save dataset
        np.savez(
            output_dir / f"worker_{worker_id}_data.npz",
            data=data,
            labels=labels,
            worker_id=worker_id
        )

    print(f"   ✓ Generated {n_workers} datasets")

    # Create manifest
    manifest = {
        "n_workers": n_workers,
        "samples_per_worker": 1000,
        "n_features": n_features,
        "task": "binary_sentiment_classification",
        "generated_with": "generate_test_data.py"
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"   ✓ Created manifest: {manifest}")


def main():
    """Generate all test data"""
    print("\n" + "="*80)
    print("GENERATING TEST DATA FOR NOOB EXAMPLES")
    print("="*80 + "\n")

    base_dir = Path(__file__).parent / "test_data"

    # Generate satellite images
    images_dir = base_dir / "satellite_images"
    generate_satellite_images(images_dir, n_images=100)

    print()

    # Generate ML datasets
    ml_dir = base_dir / "ml_datasets"
    generate_ml_datasets(ml_dir, n_workers=10)

    print("\n" + "="*80)
    print("✨ TEST DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nData saved to: {base_dir.absolute()}")
    print(f"\nYou can now run:")
    print(f"  python examples/image_processing_demo.py")
    print(f"  python examples/ml_training_demo.py")


if __name__ == "__main__":
    main()
