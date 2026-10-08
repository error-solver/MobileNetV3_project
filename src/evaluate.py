"""Evaluate a saved checkpoint on the deterministic held-out test split."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import HAM10000Dataset, get_test_transforms
from src.metrics import calculate_class_metrics, calculate_overall_metrics
from src.models import create_model
from src.utils import ensure_dir, load_checkpoint, load_metadata, split_metadata
from src.visualization import plot_confusion_matrix


def evaluate_model(model_name, checkpoint_path=None):
    output = ensure_dir(config.RESULTS_DIR / model_name)
    checkpoint_path = checkpoint_path or output / "best_model.pth"
    model = create_model(model_name, pretrained=False).to(config.DEVICE)
    checkpoint = load_checkpoint(checkpoint_path, model, config.DEVICE)
    if checkpoint["model_name"] != model_name:
        raise ValueError("Checkpoint model type does not match --model")
    _, _, test_frame = split_metadata(load_metadata())
    loader = DataLoader(HAM10000Dataset(test_frame, transform=get_test_transforms()),
                        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model.eval()
    true, predicted, image_ids = [], [], []
    with torch.inference_mode():
        offset = 0
        for images, labels in tqdm(loader):
            predictions = model(images.to(config.DEVICE)).argmax(1).cpu().tolist()
            true.extend(labels.tolist())
            predicted.extend(predictions)
            image_ids.extend(test_frame.iloc[offset:offset + len(labels)]["image_id"].tolist())
            offset += len(labels)
    overall = calculate_overall_metrics(true, predicted, config.CLASS_NAMES)
    per_class = calculate_class_metrics(true, predicted, config.CLASS_NAMES)
    pd.DataFrame([overall]).to_csv(output / "overall_metrics.csv", index=False)
    per_class.to_csv(output / "class_metrics.csv", index=False)
    pd.DataFrame({"image_id": image_ids, "true": [config.CLASS_NAMES[i] for i in true],
                  "predicted": [config.CLASS_NAMES[i] for i in predicted]}).to_csv(output / "predictions.csv", index=False)
    matrix = confusion_matrix(true, predicted, labels=range(config.NUM_CLASSES))
    pd.DataFrame(matrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).to_csv(output / "confusion_matrix.csv")
    plot_confusion_matrix(matrix, config.CLASS_NAMES, output / "confusion_matrix.png")
    print(pd.Series(overall).to_string())
    print(per_class.to_string(index=False))
    return overall, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "frequency_attention"), required=True)
    parser.add_argument("--checkpoint", help="Optional checkpoint path")
    args = parser.parse_args()
    evaluate_model(args.model, args.checkpoint)


if __name__ == "__main__":
    main()
