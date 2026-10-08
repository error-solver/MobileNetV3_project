"""Inspect metadata and save exploratory plots; run explicitly when needed."""
import argparse
from PIL import Image

from src.config import IMAGE_DIRS, RESULTS_DIR
from src.utils import ensure_dir, load_metadata
from src.visualization import plot_class_distribution, plot_sample_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    frame = load_metadata()
    print("Shape:", frame.shape)
    counts = frame["dx"].value_counts().sort_index()
    print("Class distribution:\n", counts)
    print("Class percentages:\n", (counts / len(frame) * 100).round(2))
    print("Missing values:\n", frame.isna().sum())
    output = ensure_dir(RESULTS_DIR / "analysis")
    plot_class_distribution(counts, output / "class_distribution.png")
    lookup = {path.stem: path for directory in IMAGE_DIRS for path in directory.glob("*.jpg")}
    samples = []
    for _, row in frame.sample(min(args.samples, len(frame)), random_state=42).iterrows():
        path = lookup.get(str(row["image_id"]))
        if path:
            with Image.open(path) as image:
                samples.append((image.convert("RGB").copy(), row["dx"]))
    if samples:
        plot_sample_grid(samples, output / "sample_grid.png")
        print("Sample grid saved:", output / "sample_grid.png")


if __name__ == "__main__":
    main()
