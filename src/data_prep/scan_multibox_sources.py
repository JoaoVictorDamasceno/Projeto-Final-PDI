"""
Dá uma volta pelos três datasets crus e conta quantas bboxes cada imagem
tem, sem aplicar nenhum filtro de área ou de label. A ideia é simples: em
vez de checar as imagens todas no olho pra achar quais têm mais de um
pólipo, essa varredura já separa isso pra gente e ainda gera um preview
colorido de cada caso, então sobra bem menos coisa pra olhar manualmente.
 
De onde vem a bbox em cada dataset:
 
    HyperKvasir   -> todas as entradas de bounding-boxes.json para a
                     imagem, sem filtrar por label == "polyp" e sem
                     filtro de área.
    CVC-ClinicDB  -> a máscara em PNG/Ground Truth, binarizada e
                     convertida em bbox com cv2.findContours +
                     cv2.boundingRect (a mesma lógica do
                     mask_to_bbox.py), só que sem o filtro
                     MIN_CONTOUR_AREA_PX que ele aplica — aqui todo
                     contorno vira caixa, até os de 1px².
    ETIS-Larib    -> mesma ideia da máscara que o CVC-ClinicDB.
 
Pra cada dataset sai um log .txt separado, listando TODAS as imagens com
o nome/id e quantas caixas foram encontradas (não só as que têm mais de
uma — fica o panorama completo).
 
E pras imagens com mais de uma caixa, é criada uma pasta de amostra em
samples/<dataset>/<img_id>/ com:
    01_original.<ext>  -> a imagem original, sem nada desenhado
    02_mask.<ext>       -> a máscara correspondente (nos 3 datasets,
                            até no HyperKvasir, mesmo a bbox lá vindo do
                            JSON — ajuda a comparar visualmente)
    03_boxes.png        -> a imagem original com todas as boxes
                            desenhadas, cada uma numa cor diferente (se
                            a imagem tem 4 boxes, são 4 cores distintas)
 
Uso:
    python inventario_bboxes_bruto.py \
        --hyperkvasir-dir data/raw/HyperKvasir \
        --cvc-dir data/raw/CVC-ClinicDB \
        --etis-dir data/raw/ETIS-LaribPolypDB \
        --out-logs-dir results/audit \
        --samples-dir data/samples/inventario_bruto
"""

import argparse
import colorsys
import json
import shutil
from pathlib import Path

import cv2

BINARY_THRESHOLD = 127
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".bmp")

Box = tuple[int, int, int, int]

def find_file(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"{stem}.*"))
    return matches[0] if matches else None

def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def distinct_colors(n: int) -> list[tuple[int, int, int]]:
    # Gera n cores RGB bem distribuídas e distintas visualmente
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR p/ OpenCV
    return colors

def all_boxes_from_mask(mask_path: Path) -> list[Box] | None:
    # Uma bbox por contorno externo da máscara, 100% bruto, sem filtro
    # nenhum de área. None se a máscara não abrir.
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes

def raw_boxes_from_json(entry: dict | None) -> list[Box]:
    # Todas as bboxes cruas do JSON, sem filtrar por label, sem filtrar área.
    if not entry or not entry.get("bbox"):
        return []
    boxes = []
    for box in entry["bbox"]:
        boxes.append((int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])))
    return boxes

def draw_boxes_multicolor(img, boxes: list[Box]):
    out = img.copy()
    colors = distinct_colors(len(boxes))
    for i, ((x1, y1, x2, y2), color) in enumerate(zip(boxes, colors)):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(i), (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def save_sample(dataset: str, img_id: str, img_path: Path, mask_path: Path | None,
                 boxes: list[Box], samples_root: Path) -> None:
    sample_dir = samples_root / dataset / img_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, sample_dir / f"01_original{img_path.suffix.lower()}")

    if mask_path is not None:
        shutil.copy2(mask_path, sample_dir / f"02_mask{mask_path.suffix.lower()}")

    img = cv2.imread(str(img_path))
    if img is not None:
        boxed = draw_boxes_multicolor(img, boxes)
        cv2.imwrite(str(sample_dir / "03_boxes.png"), boxed)

def write_log(out_path: Path, dataset_title: str, counts: list[tuple[str, int]],
              errors: list[tuple[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(counts)
    n_multi = sum(1 for _, n in counts if n > 1)
    n_zero = sum(1 for _, n in counts if n == 0)
    n_one = sum(1 for _, n in counts if n == 1)

    lines = []
    lines.append("=" * 70)
    lines.append(f"INVENTÁRIO BRUTO DE BBOXES — {dataset_title}")
    lines.append("=" * 70)
    lines.append(f"\nTotal de imagens verificadas: {n_total}")
    lines.append(f"  sem nenhuma bbox: {n_zero}")
    lines.append(f"  com exatamente 1 bbox: {n_one}")
    lines.append(f"  com mais de 1 bbox (foram para samples/): {n_multi}")
    if errors:
        lines.append(f"\nImagens não verificadas (erro): {len(errors)}")
        for img_id, motivo in errors:
            lines.append(f"  {img_id}: {motivo}")

    lines.append("\n" + "-" * 70)
    lines.append("DETALHE POR IMAGEM (todas, ordenadas por nome)")
    lines.append("-" * 70)
    for img_id, n in sorted(counts, key=lambda x: x[0]):
        marca = "  <- MAIS DE 1 PÓLIPO" if n > 1 else ""
        lines.append(f"  {img_id}: {n} caixa(s){marca}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def process_hyperkvasir(hyperkvasir_dir: Path, samples_root: Path, out_log: Path) -> None:
    img_dir = hyperkvasir_dir / "images"
    mask_dir = hyperkvasir_dir / "masks"
    json_path = hyperkvasir_dir / "bounding-boxes.json"

    with open(json_path) as f:
        json_data = json.load(f)

    counts = []
    errors = []

    image_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img_path in image_paths:
        img_id = img_path.stem
        boxes = raw_boxes_from_json(json_data.get(img_id))
        counts.append((img_id, len(boxes)))

        if len(boxes) > 1:
            mask_path = find_file(mask_dir, img_id)
            save_sample("hyperkvasir", img_id, img_path, mask_path, boxes, samples_root)

    write_log(out_log, "HYPERKVASIR (fonte: bounding-boxes.json)", counts, errors)
    print(f"hyperkvasir: {len(counts)} imagens, {sum(1 for _, n in counts if n > 1)} com >1 bbox")

def process_mask_based(dataset: str, img_dir: Path, mask_dir: Path,
                        samples_root: Path, out_log: Path) -> None:
    counts = []
    errors = []

    image_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    for img_path in image_paths:
        img_id = img_path.stem

        mask_path = find_file(mask_dir, img_id)
        if mask_path is None:
            errors.append((img_id, "máscara não encontrada"))
            continue

        boxes = all_boxes_from_mask(mask_path)
        if boxes is None:
            errors.append((img_id, "máscara ilegível"))
            continue

        counts.append((img_id, len(boxes)))

        if len(boxes) > 1:
            save_sample(dataset, img_id, img_path, mask_path, boxes, samples_root)

    write_log(out_log, f"{dataset.upper()} (fonte: máscara bruta, sem filtro de área)", counts, errors)
    print(f"{dataset}: {len(counts)} imagens, {sum(1 for _, n in counts if n > 1)} com >1 bbox")

def main():
    parser = argparse.ArgumentParser(
        description="Inventário bruto de bboxes nos três datasets crus (HyperKvasir, CVC-ClinicDB, ETIS-Larib)"
    )
    parser.add_argument("--hyperkvasir-dir", required=True)
    parser.add_argument("--cvc-dir", required=True)
    parser.add_argument("--etis-dir", required=True)
    parser.add_argument("--out-logs-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    args = parser.parse_args()

    hyperkvasir_dir = Path(args.hyperkvasir_dir)
    cvc_dir = Path(args.cvc_dir)
    etis_dir = Path(args.etis_dir)
    out_logs_dir = Path(args.out_logs_dir)
    samples_root = Path(args.samples_dir)

    # limpa samples de rodadas anteriores para não misturar execuções
    if samples_root.exists():
        shutil.rmtree(samples_root)

    process_hyperkvasir(
        hyperkvasir_dir, samples_root, out_logs_dir / "hyperkvasir_log.txt"
    )
    process_mask_based(
        "cvcclinicdb", cvc_dir / "PNG" / "Original", cvc_dir / "PNG" / "Ground Truth",
        samples_root, out_logs_dir / "cvcclinicdb_log.txt"
    )
    process_mask_based(
        "etislarib", etis_dir / "images", etis_dir / "masks",
        samples_root, out_logs_dir / "etislarib_log.txt"
    )

    print(f"\nLogs salvos em: {out_logs_dir}")
    print(f"Amostras (imagens com >1 bbox) salvas em: {samples_root}")

if __name__ == "__main__":
    main()