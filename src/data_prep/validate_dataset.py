"""
Validação do dataset consolidado (saída do consolidate_dataset.py).

Verifica, no diretório processado:
    - total de imagens e contagem por split (1808 = 1446/181/181, como no artigo)
    - total de caixas (--expected-boxes)
    - correspondência 1:1 entre imagens e labels em cada split
    - nenhuma imagem repetida em dois splits
    - caixas no formato YOLO (classe 0, coordenadas em [0, 1], área positiva)
    - nenhuma caixa abaixo de MIN_BOX_AREA_PX
    - existência do data.yaml

Sai com código 1 se qualquer verificação falhar.

Uso:
    python src/data_prep/validate_dataset.py --data-dir data/processed
"""

import argparse
import sys
from pathlib import Path

import cv2

from consolidate_dataset import EXPECTED_TOTAL, IMAGE_EXTENSIONS, MIN_BOX_AREA_PX

SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 1446, "val": 181, "test": 181}
EXPECTED_BOXES_DEFAULT = 1919
AREA_TOLERANCE_PX = 0.5  # as coordenadas são gravadas com 6 casas decimais


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok: bool, description: str, detail: str = "") -> None:
        status = "OK   " if ok else "FALHA"
        suffix = f" — {detail}" if detail and not ok else ""
        print(f"[{status}] {description}{suffix}")
        if not ok:
            self.failures += 1


def list_images(images_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p for p in sorted(images_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }


def list_labels(labels_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(labels_dir.glob("*.txt"))}


def parse_label(label_path: Path) -> list[tuple[float, ...]]:
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    return [tuple(map(float, l.split())) for l in lines]


def box_problems(box: tuple[float, ...]) -> str | None:
    if len(box) != 5:
        return f"esperava 5 campos, encontrou {len(box)}"
    class_id, cx, cy, bw, bh = box
    if class_id != 0:
        return f"classe {class_id:g} (esperado 0)"
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
        return "coordenada fora de [0, 1]"
    if bw <= 0 or bh <= 0:
        return "largura ou altura não positiva"
    return None


def main():
    parser = argparse.ArgumentParser(description="Validação do dataset consolidado")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--expected-boxes", type=int, default=EXPECTED_BOXES_DEFAULT)
    args = parser.parse_args()

    root = Path(args.data_dir)
    report = Report()

    images = {s: list_images(root / "images" / s) for s in SPLITS}
    labels = {s: list_labels(root / "labels" / s) for s in SPLITS}

    total_images = sum(len(v) for v in images.values())
    report.check(total_images == EXPECTED_TOTAL,
                 f"total de imagens = {EXPECTED_TOTAL}", f"encontrado {total_images}")

    for split in SPLITS:
        expected = EXPECTED_SPLIT_COUNTS[split]
        report.check(len(images[split]) == expected,
                     f"{split}: {expected} imagens", f"encontrado {len(images[split])}")

    for split in SPLITS:
        without_label = sorted(set(images[split]) - set(labels[split]))
        without_image = sorted(set(labels[split]) - set(images[split]))
        report.check(not without_label and not without_image,
                     f"{split}: imagens e labels em correspondência 1:1",
                     f"{len(without_label)} imagem(ns) sem label ({without_label[:3]}), "
                     f"{len(without_image)} label(s) sem imagem ({without_image[:3]})")

    overlaps = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlaps += [(a, b, s) for s in sorted(set(images[a]) & set(images[b]))]
    report.check(not overlaps, "nenhuma imagem em mais de um split",
                 f"{len(overlaps)} repetida(s), ex.: {overlaps[:3]}")

    n_boxes = 0
    bad_format, small_boxes = [], []
    for split in SPLITS:
        for stem, label_path in labels[split].items():
            boxes = parse_label(label_path)
            n_boxes += len(boxes)
            for i, box in enumerate(boxes):
                problem = box_problems(box)
                if problem:
                    bad_format.append(f"{split}/{stem} box {i}: {problem}")
                    continue

                img_path = images[split].get(stem)
                if img_path is None:
                    continue  # já reportado na correspondência 1:1
                img = cv2.imread(str(img_path))
                if img is None:
                    bad_format.append(f"{split}/{stem}: imagem ilegível")
                    break
                h, w = img.shape[:2]
                area = box[3] * w * box[4] * h
                if area < MIN_BOX_AREA_PX - AREA_TOLERANCE_PX:
                    small_boxes.append(f"{split}/{stem} box {i}: {area:.0f}px²")

    report.check(n_boxes == args.expected_boxes,
                 f"total de caixas = {args.expected_boxes}", f"encontrado {n_boxes}")
    report.check(not bad_format, "todas as caixas em formato YOLO válido",
                 f"{len(bad_format)} problema(s), ex.: {bad_format[:3]}")
    report.check(not small_boxes, f"nenhuma caixa abaixo de {MIN_BOX_AREA_PX}px²",
                 f"{len(small_boxes)} caixa(s), ex.: {small_boxes[:3]}")

    report.check((root / "data.yaml").exists(), "data.yaml presente")

    print()
    if report.failures:
        print(f"{report.failures} verificação(ões) falharam.")
        sys.exit(1)
    print("Dataset válido.")


if __name__ == "__main__":
    main()