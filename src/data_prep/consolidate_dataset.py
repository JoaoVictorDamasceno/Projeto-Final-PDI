"""
consolidate_dataset.py

Junta HyperKvasir, CVC-ClinicDB e ETIS-Larib num dataset único no formato
que o Ultralytics espera (images/ + labels/ com .txt normalizado).

HyperKvasir já vem com bbox pronta (bounding-boxes.json). CVC-ClinicDB e
ETIS-Larib só têm máscara, então a bbox sai do mask_to_bbox().

Split 80/10/10 por imagem, seed fixa pra dar sempre o mesmo resultado.
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from mask_to_bbox import mask_to_bbox

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASSES = ["polyp"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

# números batendo com o artigo, só pra avisar se o dataset baixado vier diferente
EXPECTED_COUNTS_BY_SOURCE = {"hyperkvasir": 1000, "cvcclinicdb": 612, "etislarib": 196}
EXPECTED_TOTAL = 1808


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    bw: float
    bh: float

    def as_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.bw:.6f} {self.bh:.6f}"

    def is_valid(self) -> bool:
        in_range = all(0.0 <= v <= 1.0 for v in (self.cx, self.cy, self.bw, self.bh))
        has_area = self.bw > 0 and self.bh > 0
        return in_range and has_area


@dataclass(frozen=True)
class Sample:
    image_path: Path
    boxes: list[YoloBox]
    source: str


def pixel_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_id: int = 0) -> YoloBox:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return YoloBox(class_id, cx, cy, bw, bh)


def load_hyperkvasir_annotations(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def hyperkvasir_boxes_for_image(annotations: dict, img_id: str, img_w: int, img_h: int) -> list[YoloBox]:
    entry = annotations.get(img_id)
    if not entry or not entry.get("bbox"):
        return []

    boxes = []
    for box in entry["bbox"]:
        yolo_box = pixel_bbox_to_yolo(box["xmin"], box["ymin"], box["xmax"], box["ymax"], img_w, img_h)
        if yolo_box.is_valid():
            boxes.append(yolo_box)
    return boxes


def collect_hyperkvasir(root: Path) -> list[Sample]:
    img_dir = root / "images"
    json_path = root / "bounding-boxes.json"

    if not img_dir.exists() or not json_path.exists():
        print(f"[aviso] HyperKvasir: 'images' ou 'bounding-boxes.json' não encontrados em {root}, pulando")
        return []

    annotations = load_hyperkvasir_annotations(json_path)
    samples = []

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[aviso] HyperKvasir: não consegui abrir {img_path}, pulando")
            continue
        h, w = img.shape[:2]

        boxes = hyperkvasir_boxes_for_image(annotations, img_path.stem, w, h)
        if boxes:
            samples.append(Sample(img_path, boxes, "hyperkvasir"))

    return samples


def find_matching_mask(img_path: Path, mask_dir: Path) -> Path | None:
    # nome igual primeiro, senão tenta achar por stem (imagem .png, máscara .tif etc)
    same_name = mask_dir / img_path.name
    if same_name.exists():
        return same_name

    candidates = list(mask_dir.glob(f"{img_path.stem}.*"))
    return candidates[0] if candidates else None


def mask_boxes_for_image(mask_path: Path) -> list[YoloBox]:
    boxes = []
    for pixel_box in mask_to_bbox(mask_path):
        yolo_box = pixel_bbox_to_yolo(
            pixel_box.x1, pixel_box.y1, pixel_box.x2, pixel_box.y2,
            pixel_box.img_w, pixel_box.img_h,
        )
        if yolo_box.is_valid():
            boxes.append(yolo_box)
    return boxes


def collect_mask_based(root: Path, source_name: str, img_folder: str, mask_folder: str) -> list[Sample]:
    img_dir = root / img_folder
    mask_dir = root / mask_folder

    if not img_dir.exists():
        print(f"[aviso] {source_name}: pasta de imagens não encontrada em {img_dir}, pulando")
        return []

    samples = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = find_matching_mask(img_path, mask_dir)
        if mask_path is None:
            continue

        boxes = mask_boxes_for_image(mask_path)
        if boxes:
            samples.append(Sample(img_path, boxes, source_name))

    return samples


def collect_all_samples(hyperkvasir_dir: Path, cvc_dir: Path, etis_dir: Path) -> dict[str, list[Sample]]:
    return {
        "hyperkvasir": collect_hyperkvasir(hyperkvasir_dir),
        "cvcclinicdb": collect_mask_based(cvc_dir, "cvcclinicdb", "PNG/Original", "PNG/Ground Truth"),
        "etislarib": collect_mask_based(etis_dir, "etislarib", "images", "masks"),
    }


def split_dataset(samples: list[Sample], ratios: dict, seed: int) -> dict[str, list[Sample]]:
    # Random local em vez de random.seed() global, pra não bagunçar estado
    # de fora se isso aqui for chamado mais de uma vez (ou em teste)
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_split(samples: list[Sample], split_name: str, out_root: Path) -> None:
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        new_stem = f"{sample.source}_{sample.image_path.stem}"  # evita colisão entre datasets
        shutil.copy2(sample.image_path, img_out / f"{new_stem}{sample.image_path.suffix.lower()}")

        lines = [box.as_line() for box in sample.boxes]
        (lbl_out / f"{new_stem}.txt").write_text("\n".join(lines))


def write_data_yaml(out_root: Path) -> Path:
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def warn_if_counts_diverge(samples_by_source: dict[str, list[Sample]]) -> None:
    for source, expected in EXPECTED_COUNTS_BY_SOURCE.items():
        actual = len(samples_by_source.get(source, []))
        if actual != expected:
            print(f"[alerta] {source}: {actual} imagens coletadas, artigo reporta {expected}")

    total = sum(len(s) for s in samples_by_source.values())
    if total != EXPECTED_TOTAL:
        print(f"[alerta] total coletado ({total}) diferente do total do artigo ({EXPECTED_TOTAL})")


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidação do dataset de pólipos")
    parser.add_argument("--hyperkvasir", required=True)
    parser.add_argument("--cvc-clinicdb", required=True)
    parser.add_argument("--etis-larib", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    samples_by_source = collect_all_samples(
        Path(args.hyperkvasir), Path(args.cvc_clinicdb), Path(args.etis_larib)
    )
    for source, samples in samples_by_source.items():
        print(f"{source}: {len(samples)} imagens coletadas")
    warn_if_counts_diverge(samples_by_source)

    all_samples = [s for source_samples in samples_by_source.values() for s in source_samples]
    if not all_samples:
        raise SystemExit("Nenhuma amostra encontrada, confere os caminhos passados")

    splits = split_dataset(all_samples, SPLIT_RATIOS, SEED)

    out_root = Path(args.out)
    for split_name, split_samples in splits.items():
        write_split(split_samples, split_name, out_root)
        print(f"{split_name}: {len(split_samples)} imagens")

    yaml_path = write_data_yaml(out_root)

    print(f"\nDataset unificado e pronto em: {out_root}")
    print(f"Arquivo de configuração para o YOLO: {yaml_path}")


if __name__ == "__main__":
    main()