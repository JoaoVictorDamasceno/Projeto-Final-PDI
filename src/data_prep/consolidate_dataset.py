"""
consolidate_dataset.py

Junta HyperKvasir, CVC-ClinicDB e ETIS-Larib num único dataset no formato
que o YOLO/Ultralytics espera (imagens + labels .txt normalizados).

HyperKvasir já vem com bbox pronta -> só converte pro formato YOLO.
CVC-ClinicDB e ETIS-Larib só têm máscara -> usa mask_to_bbox() pra gerar
a bbox a partir da máscara.

Split final: 80% treino / 10% val / 10% teste, por imagem (não por vídeo,
seguindo a mesma divisão do artigo — vale lembrar isso na discussão do
relatório, já que pode causar vazamento entre frames do mesmo vídeo).

python consolidate_dataset.py --hyperkvasir X --cvc-clinicdb Y --etis-larib Z --out data/processed
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import cv2
import yaml

from mask_to_bbox import mask_to_bbox

SEED = 42
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASSES = ["polyp"]  # classe única


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def collect_hyperkvasir(root: Path):
    # espera root/images/*.jpg e root/bboxes/*.txt (x1 y1 x2 y2 em pixels, por linha)
    samples = []
    img_dir, ann_dir = root / "images", root / "bboxes"
    if not img_dir.exists():
        print(f"[aviso] HyperKvasir não achado em {img_dir}, pulando")
        return samples

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes_yolo = []
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if ann_path.exists():
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
    # espera root/images e root/masks (mesmo nome de arquivo, extensão pode variar)
    samples = []
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.exists():
        print(f"[aviso] {source_name} não achado em {img_dir}, pulando")
        return samples

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".bmp"):
            continue

        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
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
        new_stem = f"{source}_{img_path.stem}"  # evita colisão de nome entre datasets
        shutil.copy2(img_path, img_out / f"{new_stem}{img_path.suffix.lower()}")

        lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in boxes]
        (lbl_out / f"{new_stem}.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Consolidação do dataset de pólipos")
    parser.add_argument("--hyperkvasir", required=True)
    parser.add_argument("--cvc-clinicdb", required=True)
    parser.add_argument("--etis-larib", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    all_samples = []
    all_samples += collect_hyperkvasir(Path(args.hyperkvasir))
    all_samples += collect_mask_based(Path(args.cvc_clinicdb), "cvcclinicdb")
    all_samples += collect_mask_based(Path(args.etis_larib), "etislarib")

    print(f"total de imagens coletadas: {len(all_samples)}")
    if not all_samples:
        raise SystemExit("nenhuma amostra encontrada, confere os caminhos passados")

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    out_root = Path(args.out)
    for name, samples in splits.items():
        write_split(samples, name, out_root)
        print(f"{name}: {len(samples)} imagens")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\ndataset pronto em: {out_root}")
    print(f"config: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
