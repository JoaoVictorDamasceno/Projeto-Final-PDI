"""
Consolidação de HyperKvasir, CVC-ClinicDB e ETIS-Larib em um dataset único
no formato Ultralytics (images/ e labels/ por split, mais data.yaml)

Fontes de bbox:
    HyperKvasir   -> bounding-boxes.json
    CVC / ETIS    -> derivada da máscara via mask_to_bbox()

Split 80/10/10 por imagem, com seed fixa.

Filtragem de bboxes:
    CVC / ETIS    -> nenhuma aqui; o ruído de máscara é tratado em mask_to_bbox()
    HyperKvasir   -> uma bbox 100% contida em outra bbox da mesma imagem é
                     descartada (só a menor); a imagem e as demais boxes são mantidas

Uso:
    python src/data_prep/consolidate_dataset.py \
    --hyperkvasir data/raw/HyperKvasir \
        --cvc data/raw/CVC-ClinicDB \
        --etis data/raw/ETIS-LaribPolypDB \
    --out data/processed
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from mask_to_bbox import mask_to_bbox

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASSES = ["polyp"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

PixelBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


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


@dataclass
class Audit:
    # contagens acumuladas durante a coleta, usadas no resumo final
    images_analyzed: dict[str, int] = field(default_factory=dict)   # fonte -> nº de imagens
    contained_removed: dict[str, int] = field(default_factory=dict)  # id HyperKvasir -> nº de boxes removidas


def pixel_area(x1, y1, x2, y2) -> float:
    return (x2 - x1) * (y2 - y1)


def pixel_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_id: int = 0) -> YoloBox:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return YoloBox(class_id, cx, cy, bw, bh)


def is_inside(inner: PixelBox, outer: PixelBox) -> bool:
    # 100% dentro: os quatro lados de `inner` estão dentro (ou encostados) nos de `outer`
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def drop_contained_boxes(boxes: list[PixelBox]) -> tuple[list[PixelBox], list[PixelBox]]:
    # Retorna (mantidas, descartadas). Uma box é descartada se estiver 100% dentro de
    # outra box maior; só a menor sai. Se duas boxes forem idênticas, sobra a primeira.
    kept, dropped = [], []
    for i, inner in enumerate(boxes):
        inner_area = pixel_area(*inner)
        contained = False
        for j, outer in enumerate(boxes):
            if i == j or not is_inside(inner, outer):
                continue
            outer_area = pixel_area(*outer)
            if outer_area > inner_area or (outer_area == inner_area and j < i):
                contained = True
                break
        (dropped if contained else kept).append(inner)
    return kept, dropped


def load_hyperkvasir_annotations(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def hyperkvasir_boxes_for_image(annotations: dict, img_id: str, img_w: int, img_h: int,
                                audit: Audit) -> list[YoloBox]:
    entry = annotations.get(img_id)
    if not entry or not entry.get("bbox"):
        return []

    pixel_boxes = [
        (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        for box in entry["bbox"]
        if box.get("label") == "polyp"
    ]

    kept, dropped = drop_contained_boxes(pixel_boxes)
    for box in dropped:
        print(f"[filtro] bbox contida em outra descartada em hyperkvasir/{img_id}: área={pixel_area(*box):.0f}px²")
    if dropped:
        audit.contained_removed[img_id] = len(dropped)

    boxes = []
    for x1, y1, x2, y2 in kept:
        yolo_box = pixel_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
        if yolo_box.is_valid():
            boxes.append(yolo_box)
        else:
            print(f"[aviso] bbox inválida descartada em hyperkvasir/{img_id}: ({x1}, {y1}, {x2}, {y2})")
    return boxes


def collect_hyperkvasir(root: Path, audit: Audit) -> list[Sample]:
    img_dir = root / "images"
    json_path = root / "bounding-boxes.json"

    if not img_dir.exists() or not json_path.exists():
        print(f"[aviso] HyperKvasir: 'images' ou 'bounding-boxes.json' não encontrados em {root}, pulando")
        return []

    annotations = load_hyperkvasir_annotations(json_path)
    samples = []
    analyzed = 0

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[aviso] HyperKvasir: não consegui abrir {img_path}, pulando")
            continue
        analyzed += 1
        h, w = img.shape[:2]

        boxes = hyperkvasir_boxes_for_image(annotations, img_path.stem, w, h, audit)
        if boxes:
            samples.append(Sample(img_path, boxes, "hyperkvasir"))

    audit.images_analyzed["hyperkvasir"] = analyzed
    return samples


def find_matching_mask(img_path: Path, mask_dir: Path) -> Path | None:
    same_name = mask_dir / img_path.name
    if same_name.exists():
        return same_name

    # a extensão da máscara pode diferir da imagem (ex.: .png vs .tif) 
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
        else:
            print(f"[aviso] bbox inválida descartada em {mask_path.name}: "
                  f"({pixel_box.x1}, {pixel_box.y1}, {pixel_box.x2}, {pixel_box.y2})")
    return boxes


def collect_mask_based(root: Path, source_name: str, img_folder: str, mask_folder: str,
                       audit: Audit) -> list[Sample]:
    img_dir = root / img_folder
    mask_dir = root / mask_folder

    if not img_dir.exists():
        print(f"[aviso] {source_name}: pasta de imagens não encontrada em {img_dir}, pulando")
        return []

    samples = []
    analyzed = 0
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = find_matching_mask(img_path, mask_dir)
        if mask_path is None:
            print(f"[aviso] {source_name}: máscara não encontrada para {img_path.name}, pulando")
            continue
        analyzed += 1

        boxes = mask_boxes_for_image(mask_path)
        if boxes:
            samples.append(Sample(img_path, boxes, source_name))

    audit.images_analyzed[source_name] = analyzed
    return samples


def collect_all_samples(hyperkvasir_dir: Path, cvc_dir: Path, etis_dir: Path,
                        audit: Audit) -> dict[str, list[Sample]]:
    return {
        "hyperkvasir": collect_hyperkvasir(hyperkvasir_dir, audit),
        "cvcclinicdb": collect_mask_based(cvc_dir, "cvcclinicdb", "PNG/Original", "PNG/Ground Truth", audit),
        "etislarib": collect_mask_based(etis_dir, "etislarib", "images", "masks", audit),
    }


def split_dataset(samples: list[Sample], ratios: dict, seed: int) -> dict[str, list[Sample]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled) 

    n = len(shuffled)
    # round() em vez de int(): com 1808 imagens dá 1446/181/181 (como no artigo);
    # int() daria 1446/180/182. O teste fica com o restante.
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def clean_output(out_root: Path) -> None:
    # Apaga images/ e labels/ de execuções anteriores. Sem isso, uma imagem que
    # muda de split entre execuções fica nos dois (o arquivo antigo não é removido).
    for sub in ("images", "labels"):
        shutil.rmtree(out_root / sub, ignore_errors=True)


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


def print_final_summary(samples_by_source: dict[str, list[Sample]], audit: Audit) -> None:
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)

    total_analyzed = sum(audit.images_analyzed.values())
    total_samples = sum(len(s) for s in samples_by_source.values())
    total_boxes = sum(len(sample.boxes) for s in samples_by_source.values() for sample in s)
    print(f"Imagens analisadas: {total_analyzed}")
    print(f"Imagens com ao menos 1 bbox (no dataset): {total_samples}")
    print(f"Total de bboxes: {total_boxes}")
    for source, samples in samples_by_source.items():
        n_boxes = sum(len(sample.boxes) for sample in samples)
        print(f"  {source}: {audit.images_analyzed.get(source, 0)} analisadas, "
              f"{len(samples)} no dataset, {n_boxes} bboxes")

    n_removed = sum(audit.contained_removed.values())
    print(f"\nHyperKvasir: {n_removed} bbox(es) removida(s) por estarem contidas em outra, "
          f"em {len(audit.contained_removed)} imagem(ns)")


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidação do dataset de pólipos")
    parser.add_argument("--hyperkvasir", required=True)
    parser.add_argument("--cvc-clinicdb", required=True)
    parser.add_argument("--etis-larib", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    audit = Audit()
    samples_by_source = collect_all_samples(
        Path(args.hyperkvasir), Path(args.cvc_clinicdb), Path(args.etis_larib), audit
    )
    for source, samples in samples_by_source.items():
        print(f"{source}: {len(samples)} imagens coletadas")

    all_samples = [s for source_samples in samples_by_source.values() for s in source_samples]
    if not all_samples:
        raise SystemExit("Nenhuma amostra encontrada, confere os caminhos passados")

    splits = split_dataset(all_samples, SPLIT_RATIOS, SEED)

    out_root = Path(args.out)
    clean_output(out_root)
    for split_name, split_samples in splits.items():
        write_split(split_samples, split_name, out_root)
        print(f"{split_name}: {len(split_samples)} imagens")

    yaml_path = write_data_yaml(out_root)

    print_final_summary(samples_by_source, audit)

    print(f"\nDataset unificado e pronto em: {out_root}")
    print(f"Arquivo de configuração para o YOLO: {yaml_path}")


if __name__ == "__main__":
    main()