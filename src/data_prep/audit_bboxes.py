"""
Auditoria das bboxes do dataset consolidado.

Para cada label com mais de uma bbox, verifica se a contagem é consistente
com a fonte original:

    cvcclinicdb / etislarib : recontagem das regiões da máscara bruta, sem
                              filtro de área; deve coincidir com o nº de boxes.
    hyperkvasir             : cruzamento com bounding-boxes.json (nº de
                              entradas e label == "polyp").

Veredictos por arquivo: DE_FABRICA (consistente com a fonte), DIVERGENCIA
(inconsistente, requer investigação) e ERRO (arquivo de origem não encontrado).
Boxes com área abaixo de --area-risk-threshold são sinalizadas como
heurística de possível ruído.

Saídas:
    --out-log       log de texto com resumo geral e detalhe por arquivo
    --previews-dir  imagens com boxes desenhadas, apenas para DIVERGENCIA
                    (limitado por --max-previews)

Uso:
    python src/data_prep/audit_bboxes.py \
        --labels-dir data/processed/labels \
        --images-dir data/processed/images \
        --cvc-raw data/raw/CVC-ClinicDB \
        --etis-raw data/raw/ETIS-LaribPolypDB \
        --hyperkvasir-json data/raw/HyperKvasir/bounding-boxes.json \
        --out-log results/audit/bbox_audit_log.txt \
        --previews-dir results/audit/previews
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

BINARY_THRESHOLD = 127
AREA_RISK_THRESHOLD_DEFAULT = 1000  # px², heurística, requer confirmação visual
MAX_PREVIEWS_DEFAULT = 50


@dataclass
class FileFinding:
    source: str
    split: str
    stem: str
    label_path: Path
    n_boxes_label: int
    box_areas_px: list = field(default_factory=list)
    n_regions_raw_mask: int | None = None   # apenas cvc/etis
    json_entries_ok: bool | None = None     # apenas hyperkvasir
    verdict: str = ""
    detail: str = ""


def source_and_stem_from_label(label_path: Path) -> tuple[str, str]:
    # "cvcclinicdb_547.txt" -> ("cvcclinicdb", "547")
    name = label_path.stem
    source, _, rest = name.partition("_")
    return source, rest


def read_label_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    boxes = []
    for line in lines:
        _, cx, cy, bw, bh = map(float, line.split())
        boxes.append((cx, cy, bw, bh))
    return boxes


def find_image_for_stem(images_dir: Path, split: str, source: str, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / split / f"{source}_{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_raw_file(raw_dir: Path, subfolder: str, stem: str) -> Path | None:
    folder = raw_dir / subfolder
    if not folder.exists():
        return None
    same_name_candidates = list(folder.glob(f"{stem}.*"))
    return same_name_candidates[0] if same_name_candidates else None


def count_raw_mask_regions(mask_path: Path) -> list[float]:
    # Retorna as áreas (px²) das regiões da máscara, da maior para a menor,
    # sem aplicar filtro de área mínima.
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    return areas


def audit_mask_based_file(
    finding: FileFinding, raw_dir: Path, mask_subfolder: str, area_risk_threshold: float
) -> None:
    mask_path = find_raw_file(raw_dir, mask_subfolder, finding.stem)
    if mask_path is None:
        finding.verdict = "ERRO"
        finding.detail = f"máscara raw não encontrada em {raw_dir / mask_subfolder}/{finding.stem}.*"
        return

    raw_areas = count_raw_mask_regions(mask_path)
    finding.n_regions_raw_mask = len(raw_areas)

    if len(raw_areas) == finding.n_boxes_label:
        finding.verdict = "DE_FABRICA"
        finding.detail = (
            f"máscara bruta já tem {len(raw_areas)} região(ões) desconectada(s) "
            f"sem nenhum filtro de área — bate com as {finding.n_boxes_label} boxes do label. "
            f"Fragmentação é original da anotação, não do nosso código."
        )
    else:
        finding.verdict = "DIVERGENCIA"
        finding.detail = (
            f"máscara bruta tem {len(raw_areas)} região(ões), mas o label tem "
            f"{finding.n_boxes_label} boxes — NÃO bate. Investigar mask_to_bbox.py "
            f"(possível inconsistência entre a contagem e o filtro aplicado)."
        )

    n_small = sum(1 for a in finding.box_areas_px if a < area_risk_threshold)
    if n_small:
        finding.detail += (
            f" [heurística: {n_small}/{len(finding.box_areas_px)} box(es) com área < "
            f"{area_risk_threshold}px² — possível ruído/reflexo, requer confirmação visual]"
        )


def audit_hyperkvasir_file(finding: FileFinding, json_data: dict) -> None:
    entry = json_data.get(finding.stem)
    if entry is None or not entry.get("bbox"):
        finding.verdict = "ERRO"
        finding.detail = f"entrada '{finding.stem}' não encontrada (ou sem bbox) no bounding-boxes.json"
        return

    json_boxes = entry["bbox"]
    non_polyp = [b for b in json_boxes if b.get("label") != "polyp"]
    finding.json_entries_ok = len(non_polyp) == 0

    if len(json_boxes) != finding.n_boxes_label:
        finding.verdict = "DIVERGENCIA"
        finding.detail = (
            f"JSON tem {len(json_boxes)} entrada(s), mas o label tem "
            f"{finding.n_boxes_label} boxes — NÃO bate. Investigar parser."
        )
    elif non_polyp:
        finding.verdict = "DIVERGENCIA"
        finding.detail = (
            f"{len(non_polyp)} entrada(s) no JSON NÃO têm label=='polyp' e mesmo "
            f"assim foram incluídas — o parser não está filtrando por label."
        )
    else:
        finding.verdict = "DE_FABRICA"
        finding.detail = (
            f"todas as {len(json_boxes)} entradas do JSON têm label=='polyp' — "
            f"múltiplos pólipos é anotação original do dataset, não bug nosso."
        )


def save_preview(finding: FileFinding, images_dir: Path, previews_dir: Path) -> None:
    img_path = find_image_for_stem(images_dir, finding.split, finding.source, finding.stem)
    if img_path is None:
        return

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes = read_label_boxes(finding.label_path)

    for i, (cx, cy, bw, bh) in enumerate(boxes):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        area = bw * w * bh * h
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # vermelho = discrepância real
        cv2.putText(img, f"{i}:{area:.0f}px2", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    previews_dir.mkdir(parents=True, exist_ok=True)
    out_path = previews_dir / f"{finding.source}_{finding.stem}_DIVERGENCIA.png"
    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Auditoria automática de bboxes suspeitas")
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--cvc-raw", required=True)
    parser.add_argument("--etis-raw", required=True)
    parser.add_argument("--hyperkvasir-json", required=True)
    parser.add_argument("--out-log", required=True)
    parser.add_argument("--previews-dir", required=True)
    parser.add_argument("--area-risk-threshold", type=float, default=AREA_RISK_THRESHOLD_DEFAULT)
    parser.add_argument("--max-previews", type=int, default=MAX_PREVIEWS_DEFAULT)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    cvc_raw = Path(args.cvc_raw)
    etis_raw = Path(args.etis_raw)
    previews_dir = Path(args.previews_dir)

    with open(args.hyperkvasir_json) as f:
        hyperkvasir_json = json.load(f)

    splits = ["train", "val", "test"]
    all_findings: list[FileFinding] = []
    stats = defaultdict(lambda: defaultdict(int))  # source -> verdict -> count

    for split in splits:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue

        for label_path in sorted(split_dir.glob("*.txt")):
            source, stem = source_and_stem_from_label(label_path)
            boxes = read_label_boxes(label_path)
            n_boxes = len(boxes)

            stats[source]["total_arquivos"] += 1
            if n_boxes <= 1:
                continue  # só auditamos casos com mais de 1 bbox

            img_path = find_image_for_stem(images_dir, split, source, stem)
            box_areas = []
            if img_path is not None:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                box_areas = [bw * w * bh * h for _, _, bw, bh in boxes]

            finding = FileFinding(
                source=source, split=split, stem=stem, label_path=label_path,
                n_boxes_label=n_boxes, box_areas_px=box_areas,
            )

            if source == "cvcclinicdb":
                audit_mask_based_file(finding, cvc_raw, "PNG/Ground Truth", args.area_risk_threshold)
            elif source == "etislarib":
                audit_mask_based_file(finding, etis_raw, "masks", args.area_risk_threshold)
            elif source == "hyperkvasir":
                audit_hyperkvasir_file(finding, hyperkvasir_json)
            else:
                finding.verdict = "IGNORADO"
                finding.detail = f"fonte desconhecida: {source}"

            all_findings.append(finding)
            stats[source][finding.verdict] += 1

            if finding.verdict == "DIVERGENCIA" and stats["_previews_salvas"]["total"] < args.max_previews:
                save_preview(finding, images_dir, previews_dir)
                stats["_previews_salvas"]["total"] += 1

    # monta o log
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("AUDITORIA DE BBOXES — RESUMO GERAL")
    lines.append("=" * 70)
    for source in sorted(k for k in stats if not k.startswith("_")):
        s = stats[source]
        lines.append(f"\n{source}: {s.get('total_arquivos', 0)} arquivos totais")
        lines.append(f"  DE_FABRICA (confirmado, não é bug): {s.get('DE_FABRICA', 0)}")
        lines.append(f"  DIVERGENCIA (precisa investigar):   {s.get('DIVERGENCIA', 0)}")
        lines.append(f"  ERRO (arquivo raw não encontrado):  {s.get('ERRO', 0)}")

    n_divergencias = sum(f.verdict == "DIVERGENCIA" for f in all_findings)
    lines.append(f"\nTotal de casos com mais de 1 bbox analisados: {len(all_findings)}")
    lines.append(f"Total de DIVERGENCIA (requer ação): {n_divergencias}")
    lines.append(f"Previews de DIVERGENCIA salvos em: {previews_dir} (máx {args.max_previews})")

    lines.append("\n" + "=" * 70)
    lines.append("DETALHE POR ARQUIVO")
    lines.append("=" * 70)
    for finding in sorted(all_findings, key=lambda f: (f.verdict != "DIVERGENCIA", f.source, f.stem)):
        lines.append(f"\n[{finding.verdict}] {finding.source}_{finding.stem} (split={finding.split}, "
                      f"{finding.n_boxes_label} boxes, áreas={[int(a) for a in finding.box_areas_px]}px²)")
        lines.append(f"  {finding.detail}")

    out_log.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nAuditoria concluída.")
    print(f"Log completo salvo em: {out_log}")
    print(f"Total analisado: {len(all_findings)} arquivos com >1 bbox")
    print(f"DIVERGENCIA (requer ação): {n_divergencias}")
    print(f"Previews de DIVERGENCIA salvos em: {previews_dir}")


if __name__ == "__main__":
    main()