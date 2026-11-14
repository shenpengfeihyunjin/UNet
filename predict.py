import json
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from model import ResNet34UNet
from dataset import CrackForestDataset

ROOT = Path(__file__).resolve().parent
WEIGHT_PATH = ROOT / "weights" / "best_model.pth"
OUTPUT_DIR = ROOT / "outputs" / "predictions"
MASK_DIR = OUTPUT_DIR / "masks"
COMPARE_DIR = OUTPUT_DIR / "comparisons"

THRESHOLD = 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    split = json.loads((ROOT / "split.json").read_text(encoding="utf-8"))
    val_names = split["val"]
    val_dataset = CrackForestDataset(val_names, augment=False)

    print("验证集数量:", len(val_dataset))

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    model = ResNet34UNet(pretrained=False).to(device)

    checkpoint = torch.load(
        WEIGHT_PATH,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("最佳权重加载成功")
    print("权重对应训练轮次:", checkpoint["epoch"])
    print("训练时最佳验证 Dice:", checkpoint["best_val_dice"])

    total_intersection = 0
    total_pred = 0
    total_target = 0
    results = []

    for index, name in enumerate(val_names):
        image_tensor, mask_tensor = val_dataset[index]

        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prediction = (probability >= THRESHOLD).astype(np.uint8)
        target = mask_tensor[0].numpy().astype(np.uint8)

        intersection = int((prediction * target).sum())
        pred_pixels = int(prediction.sum())
        target_pixels = int(target.sum())

        dice = (2 * intersection + 1e-7) / (
            pred_pixels + target_pixels + 1e-7
        )
        iou = (intersection + 1e-7) / (
            pred_pixels + target_pixels - intersection + 1e-7
        )

        total_intersection += intersection
        total_pred += pred_pixels
        total_target += target_pixels

        results.append({
            "image": name,
            "dice": dice,
            "iou": iou
        })

        mask_image = (prediction * 255).astype(np.uint8)
        Image.fromarray(mask_image).save(MASK_DIR / f"{name}.png")

        original = Image.open(
            ROOT / "CrackForest-dataset-master" / "image" / f"{name}.jpg"
        ).convert("RGB")
        original = np.asarray(original)

        ground_truth = (target * 255).astype(np.uint8)

        overlay = original.copy()
        overlay[prediction == 1] = (
            0.5 * overlay[prediction == 1]
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(mask_image, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Prediction")
        axes[3].imshow(overlay)
        axes[3].set_title("Prediction Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{name} | Dice: {dice:.4f} | IoU: {iou:.4f}"
        )
        plt.tight_layout()
        fig.savefig(COMPARE_DIR / f"{name}_comparison.png", dpi=150)
        plt.close(fig)

        print(
            f"[{index + 1}/{len(val_names)}] {name} "
            f"Dice: {dice:.4f} IoU: {iou:.4f}"
        )

    global_dice = (
        2 * total_intersection + 1e-7
    ) / (total_pred + total_target + 1e-7)

    global_iou = (
        total_intersection + 1e-7
    ) / (total_pred + total_target - total_intersection + 1e-7)

    csv_path = OUTPUT_DIR / "evaluation_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "dice", "iou"])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("U-Net CrackForest 验证集评价结果\n")
        f.write(f"验证图片数量: {len(val_names)}\n")
        f.write(f"最佳权重训练轮次: {checkpoint['epoch']}\n")
        f.write(f"训练记录最佳验证 Dice: {checkpoint['best_val_dice']:.4f}\n")
        f.write(f"预测阈值: {THRESHOLD}\n")
        f.write(f"全局 Dice: {global_dice:.4f}\n")
        f.write(f"全局 IoU: {global_iou:.4f}\n")
        f.write(f"逐图平均 Dice: {np.mean([r['dice'] for r in results]):.4f}\n")
        f.write(f"逐图平均 IoU: {np.mean([r['iou'] for r in results]):.4f}\n")

    print("\n预测完成！")
    print("全局 Dice:", round(global_dice, 4))
    print("全局 IoU:", round(global_iou, 4))
    print("逐图平均 Dice:", round(float(np.mean([r["dice"] for r in results])), 4))
    print("逐图平均 IoU:", round(float(np.mean([r["iou"] for r in results])), 4))
    print("预测掩膜目录:", MASK_DIR)
    print("对比图目录:", COMPARE_DIR)
    print("逐图指标:", csv_path)
    print("评价汇总:", summary_path)


if __name__ == "__main__":
    main()
