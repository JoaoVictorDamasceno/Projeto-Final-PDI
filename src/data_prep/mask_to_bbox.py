"""
Conversão de máscara de segmentação em bounding boxes

Cada região conectada da máscara (contorno externo) gera uma bbox em pixels.
Usado para CVC-ClinicDB e ETIS-Larib, que não possuem bbox nativa

Uso isolado (gera imagem de preview):
    python mask_to_bbox.py --image foo.png --mask foo_mask.png --out preview.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

BINARY_THRESHOLD = 127
MIN_CONTOUR_AREA_PX = 2  # abaixo disso é ruído de máscara


@dataclass(frozen=True)
class PixelBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    img_w: int
    img_h: int


def load_binary_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def find_polyp_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]


def mask_to_bbox(mask_path: Path, whole_mask: bool = False) -> list[PixelBBox]:
    # whole_mask=False (projeto): uma caixa por contorno externo, com filtro de área.
    # whole_mask=True (baseline): uma única caixa com os extremos de todos os
    # pixels da máscara, sem filtro, como descrito no artigo.
    binary = load_binary_mask(mask_path)
    if binary is None:
        return []

    img_h, img_w = binary.shape[:2]

    if whole_mask:
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            return []
        x1, x2 = int(xs.min()), int(xs.max()) + 1  # +1 para bater com boundingRect (x + w)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        return [PixelBBox(x1=x1, y1=y1, x2=x2, y2=y2, img_w=img_w, img_h=img_h)]

    boxes = []
    for contour in find_polyp_contours(binary):
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(PixelBBox(x1=x, y1=y, x2=x + w, y2=y + h, img_w=img_w, img_h=img_h))
    return boxes


def _preview(image_path: Path, mask_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"não consegui abrir {image_path}")

    boxes = mask_to_bbox(mask_path)
    for box in boxes:
        cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Preview salvo em {out_path} ({len(boxes)} caixa(s) encontrada(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview de bbox extraída de uma máscara")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="preview.png")
    args = parser.parse_args()

    _preview(Path(args.image), Path(args.mask), Path(args.out))