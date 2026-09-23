"""
Varredura de bboxes degeneradas no dataset consolidado.

Percorre todos os labels (train/val/test), calcula a área em pixels de cada
bbox individual e gera um log com:

    - distribuição de área por fonte (mínimo, p1, p5, mediana, máximo)
    - nº de boxes abaixo de cada threshold candidato
    - lista das boxes abaixo de --flag-threshold, com arquivo, split e área

Labels sem imagem correspondente são listados no log e ignorados.

Uso:
    python src/data_prep/scan_degenerate_boxes.py \
        --labels-dir data/processed/labels \
        --images-dir data/processed/images \
        --out-log results/audit/degenerate_boxes_log.txt \
        --flag-threshold 200
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

FLAG_THRESHOLD_DEFAULT = 200  # px² — ponto de partida para discussão, não veredito
CANDIDATE_THRESHOLDS = [50, 100, 200, 500, 1000]


def source_from_filename(filename: str) -> str:
    return filename.split("_")[0]


def find_image_for_stem(images_dir: Path, split: str, stem_with_source: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / split / f"{stem_with_source}{ext}"
        if candidate.exists():
            return candidate
    return None


def percentile(sorted_values: list, p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


def main():
    parser = argparse.ArgumentParser(description="Varredura completa de bboxes degeneradas")
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-log", required=True)
    parser.add_argument("--flag-threshold", type=float, default=FLAG_THRESHOLD_DEFAULT)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    splits = ["train", "val", "test"]

    
    areas_by_source = defaultdict(list)
    flagged = [] # (source, stem, split, box_index, area, total_boxes_no_arquivo)
    total_files = 0
    total_boxes = 0
    missing_images = []

    for split in splits:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue

        for label_path in sorted(split_dir.glob("*.txt")):
            stem_with_source = label_path.stem
            source = source_from_filename(label_path.name)

            lines = [l for l in label_path.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            total_files += 1

            img_path = find_image_for_stem(images_dir, split, stem_with_source)
            if img_path is None:
                missing_images.append(str(label_path))
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                missing_images.append(str(label_path))
                continue
            h, w = img.shape[:2]

            for i, line in enumerate(lines):
                _, cx, cy, bw, bh = map(float, line.split())
                area = bw * w * bh * h
                areas_by_source[source].append(area)
                total_boxes += 1

                if area < args.flag_threshold:
                    flagged.append((source, stem_with_source, split, i, area, len(lines)))

    # monta o log
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    lines_out = []
    lines_out.append("=" * 70)
    lines_out.append("VARREDURA COMPLETA — DISTRIBUIÇÃO DE ÁREA DE BBOXES")
    lines_out.append("=" * 70)
    lines_out.append(f"\nTotal de arquivos com bbox: {total_files}")
    lines_out.append(f"Total de boxes individuais: {total_boxes}")
    if missing_images:
        lines_out.append(f"\n[aviso] {len(missing_images)} arquivo(s) sem imagem correspondente encontrada, ignorados no cálculo de área:")
        for m in missing_images[:20]:
            lines_out.append(f"  {m}")

    lines_out.append("\n" + "-" * 70)
    lines_out.append("Distribuição de área por fonte (em px²)")
    lines_out.append("-" * 70)
    for source, areas in areas_by_source.items():
        areas_sorted = sorted(areas)
        n = len(areas_sorted)
        lines_out.append(f"\n{source}: {n} boxes")
        lines_out.append(f"  mínimo:        {areas_sorted[0]:.0f}px²")
        lines_out.append(f"  p1 (1%):       {percentile(areas_sorted, 0.01):.0f}px²")
        lines_out.append(f"  p5 (5%):       {percentile(areas_sorted, 0.05):.0f}px²")
        lines_out.append(f"  mediana (50%): {percentile(areas_sorted, 0.50):.0f}px²")
        lines_out.append(f"  máximo:        {areas_sorted[-1]:.0f}px²")

    lines_out.append("\n" + "-" * 70)
    lines_out.append("Quantos arquivos/boxes seriam afetados por diferentes thresholds de corte")
    lines_out.append("(um arquivo é 'afetado' se PELO MENOS 1 das suas boxes cai abaixo do threshold)")
    lines_out.append("-" * 70)
    
    for threshold in CANDIDATE_THRESHOLDS:
        n_boxes_below = sum(1 for areas in areas_by_source.values() for a in areas if a < threshold)
        affected_files = set()
        for source, stem, split, i, area, total_in_file in flagged:
            if area < threshold:
                affected_files.add((source, stem, split))
        pct = 100 * n_boxes_below / total_boxes if total_boxes else 0
        lines_out.append(f"  threshold < {threshold:>5}px²: {n_boxes_below} boxes ({pct:.2f}% do total)")

    lines_out.append(
        f"\n[nota] a lista detalhada de arquivos abaixo usa o --flag-threshold configurado "
        f"({args.flag_threshold}px²). Para ver o detalhe com outro threshold, rode de novo "
        f"com --flag-threshold diferente."
    )

    lines_out.append("\n" + "=" * 70)
    lines_out.append(f"ARQUIVOS COM AO MENOS 1 BOX ABAIXO DE {args.flag_threshold}px² (detalhe)")
    lines_out.append("=" * 70)
    if not flagged:
        lines_out.append(f"\nNenhum arquivo com box abaixo de {args.flag_threshold}px².")
    else:
        lines_out.append(f"\nTotal de arquivos afetados: {len(set((s, st, sp) for s, st, sp, *_ in flagged))}")
        lines_out.append(f"Total de boxes individuais abaixo do threshold: {len(flagged)}\n")
        for source, stem, split, box_idx, area, total_in_file in sorted(flagged, key=lambda x: x[4]):
            lines_out.append(
                f"  [{source}] {stem} (split={split}) — box {box_idx} de {total_in_file}: área={area:.0f}px²"
            )

    out_log.write_text("\n".join(lines_out), encoding="utf-8")

    print(f"\nVarredura concluída.")
    print(f"Total de boxes analisadas: {total_boxes}")
    print(f"Boxes abaixo de {args.flag_threshold}px²: {len(flagged)}")
    print(f"Log completo salvo em: {out_log}")


if __name__ == "__main__":
    main()