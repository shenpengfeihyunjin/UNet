from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path.home() / "experiments/neural_reproduction/UNet/CrackForest-dataset-master"
SEG_DIR = ROOT / "seg"
OUT_DIR = Path.home() / "experiments/neural_reproduction/UNet/masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

converted = 0

for seg_path in sorted(SEG_DIR.glob("*.seg")):
    width = height = None
    data_started = False
    mask = None

    with seg_path.open("r", encoding="ascii") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "width":
                width = int(parts[1])
            elif parts[0] == "height":
                height = int(parts[1])
            elif parts[0] == "data":
                if width is None or height is None:
                    raise ValueError(f"{seg_path.name}: 缺少图像尺寸")
                mask = np.zeros((height, width), dtype=np.uint8)
                data_started = True
            elif data_started:
                if len(parts) != 4:
                    raise ValueError(f"{seg_path.name}: 非法标注行 {line.strip()}")

                cls, y, x1, x2 = map(int, parts)
                if not (0 <= y < height and 0 <= x1 <= x2 < width):
                    raise ValueError(f"{seg_path.name}: 坐标越界 {line.strip()}")

                if cls == 1:
                    mask[y, x1:x2 + 1] = 255

    if mask is None:
        raise ValueError(f"{seg_path.name}: 没有找到 data 标记")

    Image.fromarray(mask).save(OUT_DIR / f"{seg_path.stem}.png")
    converted += 1

print(f"转换完成：{converted} 个掩码")
print(f"输出目录：{OUT_DIR}")
print(f"掩码尺寸：{width} x {height}")
