"""Predict one image using a trained checkpoint."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src import config
from src.dataset import get_test_transforms
from src.models import create_model
from src.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=("baseline", "frequency_attention"), default=config.MODEL_NAME)
    parser.add_argument("--checkpoint", help="Optional checkpoint path")
    args = parser.parse_args()
    path = Path(args.checkpoint) if args.checkpoint else config.RESULTS_DIR / args.model / "best_model.pth"
    model = create_model(args.model, pretrained=False).to(config.DEVICE)
    checkpoint = load_checkpoint(path, model, config.DEVICE)
    if checkpoint["model_name"] != args.model:
        raise ValueError("Checkpoint model type does not match --model")
    model.eval()
    with Image.open(args.image) as image:
        tensor = get_test_transforms()(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    winner = int(probabilities.argmax())
    print(f"Predicted class: {config.CLASS_NAMES[winner]}")
    for name, probability in zip(config.CLASS_NAMES, probabilities.tolist()):
        print(f"{name}: {probability:.4%}")
    print("Research and educational use only. This is NOT a medical diagnosis tool.")


if __name__ == "__main__":
    main()
