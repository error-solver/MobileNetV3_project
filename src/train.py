"""Train either model. No work happens until main is called."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import HAM10000Dataset, get_test_transforms, get_train_transforms
from src.models import create_model
from src.utils import ensure_dir, load_metadata, save_checkpoint, set_seed, split_metadata
from src.visualization import plot_accuracy, plot_loss


def make_debug_subset(frame, per_class=8):
    return frame.groupby("dx", group_keys=False).head(per_class).reset_index(drop=True)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "frequency_attention"), default=config.MODEL_NAME)
    parser.add_argument("--fast-debug", action="store_true", help="Use a small balanced subset and two epochs")
    args = parser.parse_args()
    set_seed()
    train_frame, val_frame, _ = split_metadata(load_metadata())
    debug = config.FAST_DEBUG or args.fast_debug
    if debug:
        train_frame = make_debug_subset(train_frame)
        val_frame = make_debug_subset(val_frame, per_class=2)
    train_dataset = HAM10000Dataset(train_frame, transform=get_train_transforms())
    val_dataset = HAM10000Dataset(val_frame, transform=get_test_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS)
    # Pretrained=True lets torchvision fetch ImageNet weights on first training run.
    model = create_model(args.model, pretrained=True).to(config.DEVICE)
    counts = train_frame["dx"].value_counts().reindex(config.CLASS_NAMES, fill_value=0)
    if (counts == 0).any():
        raise ValueError("Each class needs at least one training example")
    weights = torch.tensor(len(train_frame) / (config.NUM_CLASSES * counts.to_numpy()),
                           dtype=torch.float32, device=config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    output = ensure_dir(config.RESULTS_DIR / args.model)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}
    best_loss, stale = float("inf"), 0
    for epoch in range(1, (2 if debug else config.EPOCHS) + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, config.DEVICE, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_loss)
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_accuracy),
                           ("val_loss", val_loss), ("val_accuracy", val_accuracy)):
            history[key].append(value)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            save_checkpoint(output / "best_model.pth", model, args.model, epoch, best_loss)
        else:
            stale += 1
            if stale >= config.PATIENCE:
                print("Early stopping")
                break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    plot_loss(history, output / "loss.png")
    plot_accuracy(history, output / "accuracy.png")


if __name__ == "__main__":
    main()
