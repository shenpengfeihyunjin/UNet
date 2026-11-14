import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ResNet34UNet
from dataset import get_dataloaders


ROOT = Path(__file__).resolve().parent
WEIGHT_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"

EPOCHS = 50
BATCH_SIZE = 2
PATIENCE = 10

ENCODER_LR = 1e-5
DECODER_LR = 1e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dice


def calculate_metrics(logits, targets):
    predictions = torch.sigmoid(logits) > 0.5
    targets = targets > 0.5

    predictions = predictions.flatten()
    targets = targets.flatten()

    intersection = (predictions & targets).sum().item()
    union = (predictions | targets).sum().item()
    total_pred = predictions.sum().item()
    total_true = targets.sum().item()

    dice = (2.0 * intersection + 1e-7) / (
        total_pred + total_true + 1e-7
    )
    iou = (intersection + 1e-7) / (union + 1e-7)

    return dice, iou


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice_intersection = 0.0
    total_dice_denominator = 0.0
    total_intersection = 0.0
    total_union = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.sigmoid(logits) > 0.5
        targets = masks > 0.5

        intersection = (predictions & targets).sum().item()
        union = (predictions | targets).sum().item()

        total_dice_intersection += 2.0 * intersection
        total_dice_denominator += (
            predictions.sum().item() + targets.sum().item()
        )
        total_intersection += intersection
        total_union += union

    count = len(loader.dataset)
    average_loss = total_loss / count
    dice = (
        total_dice_intersection + 1e-7
    ) / (total_dice_denominator + 1e-7)
    iou = (
        total_intersection + 1e-7
    ) / (total_union + 1e-7)

    return average_loss, dice, iou


def main():
    set_seed()

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("训练样本:", len(train_loader.dataset))
    print("验证样本:", len(val_loader.dataset))

    model = ResNet34UNet(pretrained=True).to(device)

    encoder_params = (
        list(model.conv1.parameters())
        + list(model.bn1.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )

    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [
        p for p in model.parameters()
        if id(p) not in encoder_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR},
            {"params": decoder_params, "lr": DECODER_LR},
        ],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = combined_loss

    best_dice = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        with torch.no_grad():
            val_loss, val_dice, val_iou = run_epoch(
                model, val_loader, criterion, device
            )

        scheduler.step(val_dice)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "train_iou": train_iou,
            "val_iou": val_iou,
        }
        history.append(record)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Train Dice: {train_dice:.4f} "
            f"Val Dice: {val_dice:.4f} "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_dice,
                },
                WEIGHT_DIR / "best_model.pth"
            )
            print("已保存当前最佳模型。")
        else:
            no_improve += 1

        with open(OUTPUT_DIR / "training_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if no_improve >= PATIENCE:
            print("验证 Dice 连续多轮没有提升，提前停止训练。")
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_dice,
        },
        WEIGHT_DIR / "last_model.pth"
    )

    epochs = [item["epoch"] for item in history]

    plt.figure()
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [item["train_dice"] for item in history], label="Train Dice")
    plt.plot(epochs, [item["val_dice"] for item in history], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "dice_curve.png", dpi=200)
    plt.close()

    print("\n训练结束。")
    print("最佳验证 Dice:", round(best_dice, 4))
    print("最佳权重:", WEIGHT_DIR / "best_model.pth")
    print("训练日志:", OUTPUT_DIR / "training_log.csv")
    print("损失曲线:", OUTPUT_DIR / "loss_curve.png")
    print("Dice 曲线:", OUTPUT_DIR / "dice_curve.png")


if __name__ == "__main__":
    main()
