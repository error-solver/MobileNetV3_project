"""Shared reproducibility, data split, and checkpoint helpers."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config import CLASS_NAMES, METADATA_PATH, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path, model, model_name, epoch, best_val_loss):
    ensure_dir(Path(path).parent)
    torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                "epoch": epoch, "best_val_loss": best_val_loss,
                "class_names": list(CLASS_NAMES)}, path)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order differs from this project")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Place HAM10000 metadata at {METADATA_PATH}")
    frame = pd.read_csv(METADATA_PATH)
    required = {"image_id", "dx"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}")
    frame = frame.dropna(subset=["image_id", "dx"]).copy()
    unknown = set(frame["dx"]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown lesion classes: {sorted(unknown)}")
    return frame


def split_metadata(frame, seed=RANDOM_SEED):
    """Deterministic 70/15/15 stratified split, shared by train and evaluate."""
    train, remainder = train_test_split(frame, test_size=0.30, stratify=frame["dx"], random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.50,
                                        stratify=remainder["dx"], random_state=seed)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
