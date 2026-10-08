"""Central project settings. Override paths and settings here if needed."""
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "HAM10000"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = (DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2")
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
PATIENCE = 3
RANDOM_SEED = 42
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "frequency_attention"
FAST_DEBUG = False
