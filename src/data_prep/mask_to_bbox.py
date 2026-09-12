"""
mask_to_bbox.py

Deriva bounding boxes a partir de máscaras de segmentação binárias (usado
para CVC-ClinicDB e ETIS-Larib, que não trazem bbox pronta, só máscara).

A ideia é pegar os contornos da máscara e usar o retângulo que envolve cada
um deles (coordenadas extremas), igual descrito na seção 3.1 do artigo.

Fica separado do resto da preparação do dataset de propósito: dá pra testar
essa função sozinha em algumas imagens antes de rodar em cima do dataset
inteiro (menos chance de perder tempo descobrindo erro de anotação só depois
do dataset todo processado).
"""

import argparse
from pathlib import Path

import cv2

MIN_CONTOUR_AREA = 20  # ignora manchas minúsculas / ruído da máscara


def mask_to_bbox(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape[:2]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append((x, y, x + bw, y + bh, w, h))
    return boxes


def _preview(image_path: Path, mask_path: Path, out_path: Path):
    """Desenha as caixas encontradas em cima da imagem original, só pra
    conferir visualmente se a conversão faz sentido antes de confiar nela."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"não consegui abrir {image_path}")

    for x1, y1, x2, y2, _, _ in mask_to_bbox(mask_path):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    # uso rápido pra testar em uma imagem antes de rodar em massa:
    #   python mask_to_bbox.py --image caminho/img.png --mask caminho/mask.png --out preview.png
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="preview.png")
    args = parser.parse_args()

    _preview(Path(args.image), Path(args.mask), Path(args.out))
