import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path.home() / "experiments/neural_reproduction/UNet"
IMAGE_DIR = ROOT / "CrackForest-dataset-master" / "image"
MASK_DIR = ROOT / "masks"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CrackForestDataset(Dataset):
    def __init__(self, names, augment=False):
        self.names = names
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image = Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB")
        mask = Image.open(MASK_DIR / f"{name}.png").convert("L")

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())

        mask = np.asarray(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask[None, :, :].copy())

        return image, mask


def get_dataloaders(batch_size=2):
    split = json.loads((ROOT / "split.json").read_text(encoding="utf-8"))

    train_set = CrackForestDataset(split["train"], augment=True)
    val_set = CrackForestDataset(split["val"], augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    images, masks = next(iter(train_loader))

    print("训练集数量:", len(train_loader.dataset))
    print("验证集数量:", len(val_loader.dataset))
    print("图片张量尺寸:", tuple(images.shape))
    print("掩码张量尺寸:", tuple(masks.shape))
    print("掩码取值:", torch.unique(masks).tolist())
