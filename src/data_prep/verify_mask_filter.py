"""
Verificação do filtro de ruído do mask_to_bbox (MIN_CONTOUR_AREA_PX) nos
datasets CVC-ClinicDB e ETIS-Larib.

Percorre todas as imagens dos dois datasets, gera as bboxes com mask_to_bbox()
(sem nenhum filtro extra) e mostra o progresso no terminal. Para as imagens do
CVC que foram anotadas manualmente como ruído, salva em --samples-dir:

    <samples-dir>/cvcclinicdb/<id>/
        01_original.<ext>   imagem original
        02_mask.<ext>       máscara
        03_boxes.png        original com as bboxes (uma cor por box)

No final imprime quantas boxes cada uma dessas imagens tem. Espera-se 1 em
todas: qualquer valor diferente é sinalizado.

Uso (a partir da raiz do projeto):
    python src/data_prep/verify_mask_filter.py \
        --cvc-dir data/raw/CVC-ClinicDB \
        --etis-dir data/raw/ETIS-LaribPolypDB \
        --samples-dir data/samples/mask_to_bbox_filtrado
"""

import argparse
import shutil
import sys
from pathlib import Path

from mask_to_bbox import MIN_CONTOUR_AREA_PX, load_binary_mask, mask_to_bbox
from scan_multibox_sources import IMAGE_EXTENSIONS, find_file, save_sample

CVC_NOISE_IDS = [
    "88", "585", "575", "541", "525", "495", "470", "428", "419", "353", "345", "314",
    "270", "258", "241", "231", "213", "214", "201", "184", "168", "14", "118",
]


def collect_images(dataset_dirs: dict) -> list[tuple[str, Path, Path]]:
    # (dataset, pasta_de_máscaras, caminho_da_imagem)
    items = []
    for dataset, (img_dir, mask_dir) in dataset_dirs.items():
        if not img_dir.exists():
            print(f"[aviso] {dataset}: pasta de imagens não encontrada em {img_dir}")
            continue
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((dataset, mask_dir, p))
    return items


def main():
    parser = argparse.ArgumentParser(description="Verificação do filtro do mask_to_bbox")
    parser.add_argument("--cvc-dir", required=True)
    parser.add_argument("--etis-dir", required=True)
    parser.add_argument("--samples-dir", default="data/samples/mask_to_bbox_filtrado")
    args = parser.parse_args()

    cvc, etis = Path(args.cvc_dir), Path(args.etis_dir)
    samples_root = Path(args.samples_dir)
    dataset_dirs = {
        "cvcclinicdb": (cvc / "PNG" / "Original", cvc / "PNG" / "Ground Truth"),
        "etislarib": (etis / "images", etis / "masks"),
    }

    # limpa só a subpasta desta verificação, nunca data/samples inteiro
    if samples_root.exists():
        shutil.rmtree(samples_root)

    items = collect_images(dataset_dirs)
    total = len(items)
    print(f"MIN_CONTOUR_AREA_PX = {MIN_CONTOUR_AREA_PX} | {total} imagens para analisar")

    counts: dict[str, dict[str, int]] = {d: {} for d in dataset_dirs}
    errors = []

    for n, (dataset, mask_dir, img_path) in enumerate(items, 1):
        img_id = img_path.stem
        mask_path = find_file(mask_dir, img_id)
        if mask_path is None or load_binary_mask(mask_path) is None:
            errors.append(f"{dataset}/{img_id}")
        else:
            boxes = [(b.x1, b.y1, b.x2, b.y2) for b in mask_to_bbox(mask_path)]
            counts[dataset][img_id] = len(boxes)
            if dataset == "cvcclinicdb" and img_id in CVC_NOISE_IDS:
                save_sample(dataset, img_id, img_path, mask_path, boxes, samples_root)

        sys.stdout.write(f"\rImagens analisadas: {n}/{total}")
        sys.stdout.flush()
    print()

    print("\nResumo por dataset")
    for dataset, per_img in counts.items():
        multi = sum(1 for c in per_img.values() if c > 1)
        print(f"  {dataset}: {len(per_img)} imagens, {sum(per_img.values())} boxes, "
              f"{multi} com mais de 1 box")
    if errors:
        print(f"\n[aviso] {len(errors)} imagem(ns) sem máscara legível: {errors[:10]}")

    print("\nImagens do CVC anotadas como ruído (esperado: 1 box)")
    problems = 0
    for img_id in CVC_NOISE_IDS:
        n_boxes = counts["cvcclinicdb"].get(img_id)
        if n_boxes is None:
            print(f"  {img_id}: NÃO ENCONTRADA")
            problems += 1
        elif n_boxes != 1:
            print(f"  {img_id}: {n_boxes} boxes  <- VERIFICAR")
            problems += 1
        else:
            print(f"  {img_id}: 1 box")

    print(f"\n{len(CVC_NOISE_IDS) - problems}/{len(CVC_NOISE_IDS)} com 1 box.")
    print(f"Amostras salvas em: {samples_root}")


if __name__ == "__main__":
    main()