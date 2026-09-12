"""
01_prepare_dataset.py
======================
Consolida HyperKvasir, CVC-ClinicDB e ETIS-Larib Polyp DB em um único dataset
no formato YOLO (imagens + labels .txt normalizados), conforme Seção 3.1 do artigo.

- HyperKvasir: já possui bounding boxes -> convertidas diretamente para o formato YOLO.
- CVC-ClinicDB e ETIS-Larib: possuem apenas máscaras de segmentação binárias -> as
  bounding boxes são derivadas das coordenadas extremas (min/max linha e coluna)
  de cada máscara, como descrito no artigo.

Divisão final: 80% treino / 10% validação / 10% teste, particionada ao nível de imagem.

Uso:
    python 01_prepare_dataset.py \
        --hyperkvasir /path/to/hyperkvasir \
        --cvc-clinicdb /path/to/cvc_clinicdb \
        --etis-larib /path/to/etis_larib \
        --out /home/claude/polyp_yolo_quant/data/polyp_dataset
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASS_NAMES = ["polyp"]  # detecção de classe única, conforme o artigo


def mask_to_bbox(mask_path: Path):
    """Deriva uma (ou mais) bounding box(es) a partir das coordenadas extremas
    de uma máscara binária de segmentação, conforme Seção 3.1."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = mask.shape[:2]
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:  # descarta ruído ínfimo
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append((x, y, x + bw, y + bh, w, h))
    return boxes


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def collect_hyperkvasir(root: Path):
    """Espera estrutura: root/images/*.jpg e root/bboxes/*.txt|json com
    caixas já fornecidas pelo dataset (conforme descrito no artigo)."""
    samples = []
    img_dir = root / "images"
    ann_dir = root / "bboxes"
    if not img_dir.exists():
        print(f"[aviso] HyperKvasir não encontrado em {img_dir}, pulando.")
        return samples

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        ann_path = ann_dir / f"{img_path.stem}.txt"
        boxes_yolo = []
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if ann_path.exists():
            # formato esperado: x1 y1 x2 y2 (pixels) por linha
            for line in ann_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = map(float, parts[:4])
                boxes_yolo.append((0, *xyxy_to_yolo(x1, y1, x2, y2, w, h)))

        if boxes_yolo:
            samples.append((img_path, boxes_yolo, "hyperkvasir"))
    return samples


def collect_mask_based(root: Path, source_name: str):
    """Para CVC-ClinicDB e ETIS-Larib: imagens em root/images, máscaras em
    root/masks, boxes derivadas via mask_to_bbox()."""
    samples = []
    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.exists():
        print(f"[aviso] {source_name} não encontrado em {img_dir}, pulando.")
        return samples

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".bmp"):
            continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # tenta extensões alternativas
            candidates = list(mask_dir.glob(f"{img_path.stem}.*"))
            if not candidates:
                continue
            mask_path = candidates[0]

        boxes_px = mask_to_bbox(mask_path)
        if not boxes_px:
            continue
        boxes_yolo = [
            (0, *xyxy_to_yolo(x1, y1, x2, y2, w, h)) for (x1, y1, x2, y2, w, h) in boxes_px
        ]
        samples.append((img_path, boxes_yolo, source_name))
    return samples


def write_split(samples, split_name: str, out_root: Path):
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, boxes, source in samples:
        # nome único para evitar colisão entre datasets
        new_stem = f"{source}_{img_path.stem}"
        dst_img = img_out / f"{new_stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_img)

        lbl_lines = [
            f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in boxes
        ]
        (lbl_out / f"{new_stem}.txt").write_text("\n".join(lbl_lines))


def main():
    parser = argparse.ArgumentParser(description="Consolidação do dataset de pólipos")
    parser.add_argument("--hyperkvasir", type=str, required=True)
    parser.add_argument("--cvc-clinicdb", type=str, required=True)
    parser.add_argument("--etis-larib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_samples = []
    all_samples += collect_hyperkvasir(Path(args.hyperkvasir))
    all_samples += collect_mask_based(Path(args.cvc_clinicdb), "cvcclinicdb")
    all_samples += collect_mask_based(Path(args.etis_larib), "etislarib")

    print(f"Total de imagens únicas coletadas: {len(all_samples)}")
    if not all_samples:
        raise SystemExit("Nenhuma amostra coletada. Verifique os caminhos informados.")

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    out_root = Path(args.out)
    for split_name, split_samples in splits.items():
        write_split(split_samples, split_name, out_root)
        print(f"{split_name}: {len(split_samples)} imagens")

    # data.yaml no formato Ultralytics
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\nDataset consolidado em: {out_root}")
    print(f"Arquivo de configuração: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
