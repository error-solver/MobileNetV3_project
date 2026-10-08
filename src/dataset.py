"""Metadata-backed image dataset and transforms."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CLASS_NAMES, IMAGE_DIRS, IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, image_dirs=IMAGE_DIRS, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.image_paths = {}
        for directory in image_dirs:
            directory = Path(directory)
            if not directory.is_dir():
                raise FileNotFoundError(f"Image directory missing: {directory}")
            for path in directory.glob("*.jpg"):
                self.image_paths[path.stem] = path
        missing = set(self.dataframe["image_id"].astype(str)) - self.image_paths.keys()
        if missing:
            raise FileNotFoundError(f"Images missing for {len(missing)} metadata IDs; first: {sorted(missing)[0]}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(self.image_paths[str(row["image_id"])]) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, self.class_to_index[row["dx"]]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
