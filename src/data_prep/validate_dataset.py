"""
Validação do dataset consolidado (saída do consolidate_dataset.py), contra a
contagem oficial de cada modo.

Verifica, no diretório processado:
    - o modo gravado no dataset (mode.txt) bate com a flag passada
    - total de imagens (1808), por split (1446/181/181) e por fonte
      (HyperKvasir 1000, CVC-ClinicDB 612, ETIS-Larib 196)
    - total de caixas e por fonte, conforme o modo:
          projeto : 1917 (1063 / 646 / 208)
          baseline: 1879 (1071 / 612 / 196)
    - correspondência 1:1 entre imagens e labels em cada split
    - nenhuma imagem repetida em dois splits
    - todas as imagens legíveis
    - caixas no formato YOLO (classe 0, coordenadas em [0, 1], área positiva)
    - projeto : HyperKvasir sem nenhuma caixa 100% dentro de outra
    - baseline: CVC e ETIS com exatamente 1 caixa por imagem
    - imagens do HyperKvasir marcadas como ruído com o nº de caixas esperado no modo
    - existência do data.yaml

Sai com código 1 se qualquer verificação falhar.

Uso:
    python src/data_prep/validate_dataset.py --data-dir data/processed/projeto
    python src/data_prep/validate_dataset.py --data-dir data/processed/baseline --baseline
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

from consolidate_dataset import IMAGE_EXTENSIONS, MODE_MARKER

SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 1446, "val": 181, "test": 181}

EXPECTED_IMAGES_BY_SOURCE = {"hyperkvasir": 1000, "cvcclinicdb": 612, "etislarib": 196}
EXPECTED_TOTAL_IMAGES = sum(EXPECTED_IMAGES_BY_SOURCE.values())

# imagens do HyperKvasir marcadas na revisão manual como ruído -> nº de pólipos
# reais (modo projeto, depois que a box de ruído contida em outra é removida).
HYPERKVASIR_NOISE_EXPECTED_BOXES = {
    "101a484a-5a31-493d-97a9-3ec5650c8bb1": 1,
    "73d8b58d-6320-43c6-8bec-3ed3e37f9269": 1,
    "74138051-ac26-41c5-9bb9-c174f97c8434": 1,
    "7e340833-b661-4357-a099-b8cd65c5e3f5": 1,
    "8ea26706-d1f2-441c-93d1-0504fea75f06": 1,
    "90d30949-a733-4e0e-8fdc-b8687e8548ad": 2,  # 3 boxes no JSON: 2 pólipos reais + 1 ruído
    "dc094129-83d2-4f78-a0d0-30fa97012255": 1,
    "fff2bc86-e6ad-4ce5-989a-ab63a1c096b3": 1,
}
# baseline: sem o filtro de contenção, cada uma dessas imagens volta a ter a box de ruído (+1)
BASELINE_NOISE_EXPECTED_BOXES = {k: v + 1 for k, v in HYPERKVASIR_NOISE_EXPECTED_BOXES.items()}

# Contagem oficial por modo. Origem de cada número:
#   projeto
#     HyperKvasir  1071 boxes no JSON - 8 contidas em outra (ruído) = 1063
#     CVC-ClinicDB  670 contornos na máscara - 24 de ruído (mask_to_bbox) = 646
#     ETIS-Larib    208 contornos na máscara, nenhum removido
#   baseline
#     HyperKvasir  1071 boxes do JSON, sem filtro
#     CVC / ETIS   1 caixa por imagem (extremos da máscara inteira): 612 e 196
MODES = {
    "projeto": {
        "boxes": {"hyperkvasir": 1063, "cvcclinicdb": 646, "etislarib": 208},
        "noise": HYPERKVASIR_NOISE_EXPECTED_BOXES,
        "check_nested": True,
        "one_box_sources": (),
    },
    "baseline": {
        "boxes": {"hyperkvasir": 1071, "cvcclinicdb": 612, "etislarib": 196},
        "noise": BASELINE_NOISE_EXPECTED_BOXES,
        "check_nested": False,
        "one_box_sources": ("cvcclinicdb", "etislarib"),
    },
}

NESTED_TOLERANCE = 1e-5  # as coordenadas são gravadas com 6 casas decimais


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


def source_of(stem: str) -> str:
    # os arquivos do consolidate se chamam <fonte>_<id original>
    return stem.split("_", 1)[0]


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


def contained_pairs(boxes: list[tuple[float, ...]]) -> list[tuple[int, int]]:
    # pares (i, j) em que a box i está 100% dentro da box j (com tolerância de arredondamento)
    xyxy = [(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2) for _, cx, cy, bw, bh in boxes]
    tol = NESTED_TOLERANCE
    pairs = []
    for i, a in enumerate(xyxy):
        for j, b in enumerate(xyxy):
            if i != j and (a[0] >= b[0] - tol and a[1] >= b[1] - tol
                           and a[2] <= b[2] + tol and a[3] <= b[3] + tol):
                pairs.append((i, j))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validação do dataset consolidado")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline", action="store_true",
                        help="valida contra a contagem do baseline (default: projeto)")
    args = parser.parse_args()

    mode = "baseline" if args.baseline else "projeto"
    cfg = MODES[mode]
    expected_boxes = cfg["boxes"]
    expected_total_boxes = sum(expected_boxes.values())
    noise_expected = cfg["noise"]

    root = Path(args.data_dir)
    report = Report()
    print(f"=== MODO: {mode} ===")

    # ---- modo gravado no dataset
    marker = root / MODE_MARKER
    found_mode = marker.read_text().strip() if marker.exists() else None
    report.check(found_mode == mode, f"dataset gerado no modo '{mode}'",
                 f"marcador do dataset: {found_mode!r}")

    images = {s: list_images(root / "images" / s) for s in SPLITS}
    labels = {s: list_labels(root / "labels" / s) for s in SPLITS}

    # ---- imagens
    total_images = sum(len(v) for v in images.values())
    report.check(total_images == EXPECTED_TOTAL_IMAGES,
                 f"total de imagens = {EXPECTED_TOTAL_IMAGES}", f"encontrado {total_images}")

    for split in SPLITS:
        expected = EXPECTED_SPLIT_COUNTS[split]
        report.check(len(images[split]) == expected,
                     f"{split}: {expected} imagens", f"encontrado {len(images[split])}")

    images_by_source = Counter(source_of(stem) for s in SPLITS for stem in images[s])
    for source, expected in EXPECTED_IMAGES_BY_SOURCE.items():
        report.check(images_by_source[source] == expected,
                     f"{source}: {expected} imagens", f"encontrado {images_by_source[source]}")
    unknown = sorted(set(images_by_source) - set(EXPECTED_IMAGES_BY_SOURCE))
    report.check(not unknown, "todas as imagens pertencem a uma fonte conhecida",
                 f"fontes desconhecidas: {unknown}")

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

    unreadable = [f"{s}/{stem}" for s in SPLITS
                  for stem, path in images[s].items() if cv2.imread(str(path)) is None]
    report.check(not unreadable, "todas as imagens legíveis",
                 f"{len(unreadable)} ilegível(is), ex.: {unreadable[:3]}")

    # ---- caixas
    boxes_by_source, boxes_by_split = Counter(), Counter()
    box_count = {}  # stem -> nº de caixas
    bad_format, nested = [], []
    for split in SPLITS:
        for stem, label_path in labels[split].items():
            boxes = parse_label(label_path)
            box_count[stem] = len(boxes)
            boxes_by_source[source_of(stem)] += len(boxes)
            boxes_by_split[split] += len(boxes)

            file_ok = True
            for i, box in enumerate(boxes):
                problem = box_problems(box)
                if problem:
                    bad_format.append(f"{split}/{stem} box {i}: {problem}")
                    file_ok = False

            # o filtro de contenção do consolidate só vale para o HyperKvasir (modo projeto)
            if file_ok and source_of(stem) == "hyperkvasir":
                nested += [f"{split}/{stem}: box {i} dentro da box {j}"
                           for i, j in contained_pairs(boxes)]

    total_boxes = sum(boxes_by_source.values())
    report.check(total_boxes == expected_total_boxes,
                 f"total de caixas = {expected_total_boxes}", f"encontrado {total_boxes}")
    for source, expected in expected_boxes.items():
        report.check(boxes_by_source[source] == expected,
                     f"{source}: {expected} caixas", f"encontrado {boxes_by_source[source]}")
    report.check(not bad_format, "todas as caixas em formato YOLO válido",
                 f"{len(bad_format)} problema(s), ex.: {bad_format[:3]}")

    if cfg["check_nested"]:
        report.check(not nested, "HyperKvasir: nenhuma caixa 100% dentro de outra",
                     f"{len(nested)} caso(s), ex.: {nested[:3]}")

    for source in cfg["one_box_sources"]:
        wrong = [f"{stem}: {n}" for stem, n in box_count.items()
                 if source_of(stem) == source and n != 1]
        report.check(not wrong, f"{source}: exatamente 1 caixa por imagem",
                     f"{len(wrong)} imagem(ns), ex.: {wrong[:3]}")

    # ---- imagens de ruído do HyperKvasir
    noise_rows = [(img_id, expected, box_count.get(f"hyperkvasir_{img_id}"))
                  for img_id, expected in noise_expected.items()]
    noise_wrong = [f"{img_id}: esperado {expected}, encontrado {'ausente' if found is None else found}"
                   for img_id, expected, found in noise_rows if found != expected]
    noise_expected_total = sum(noise_expected.values())
    report.check(not noise_wrong,
                 f"imagens de ruído do HyperKvasir com as caixas esperadas no modo {mode} "
                 f"({len(noise_rows)} imagens, {noise_expected_total} caixas)",
                 f"{len(noise_wrong)} divergência(s), ex.: {noise_wrong[:3]}")

    report.check((root / "data.yaml").exists(), "data.yaml presente")

    # ---- contagem observada (informativa)
    print("\nContagem observada")
    for source in EXPECTED_IMAGES_BY_SOURCE:
        print(f"  {source}: {images_by_source[source]} imagens, {boxes_by_source[source]} caixas")
    for split in SPLITS:
        print(f"  {split}: {len(images[split])} imagens, {boxes_by_split[split]} caixas")
    print(f"  total: {total_images} imagens, {total_boxes} caixas")

    print(f"\nImagens de ruído do HyperKvasir (caixas no modo {mode}: esperado / encontrado)")
    for img_id, expected, found in noise_rows:
        print(f"  {img_id}: {expected} / {'ausente' if found is None else found}")
    found_total = sum(f for _, _, f in noise_rows if f is not None)
    print(f"  total: {noise_expected_total} / {found_total}")

    print()
    if report.failures:
        print(f"{report.failures} verificação(ões) falharam.")
        sys.exit(1)
    print(f"Dataset válido (modo {mode}).")


if __name__ == "__main__":
    main()