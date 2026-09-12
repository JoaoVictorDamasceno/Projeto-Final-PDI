"""
02_train_models.py
====================
Treina (fine-tuning a partir de pesos pré-treinados em COCO) as variantes
nano (n), medium (m) e extra large (xl) dos modelos YOLOv8, YOLOv9 e YOLOv11,
conforme Seção 3.2 e 3.3 do artigo.

Hiperparâmetros usados no artigo:
    - imgsz: 640x640
    - epochs: 100
    - batch size: 16
    - otimizador: Adam, lr0 = 1e-3
    - scheduler: cosine

Observação sobre nomenclatura do YOLOv9 (nota de rodapé do artigo):
    O YOLOv9 não usa a nomenclatura n/m/xl. As variantes de complexidade
    equivalente utilizadas são: YOLOv9t (~nano), YOLOv9m (~medium) e
    YOLOv9e (~extra large).

Requer: pip install ultralytics --break-system-packages
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

# Mapeamento arquitetura/variante -> arquivo de pesos base (pré-treinado em COCO)
MODEL_MAP = {
    ("yolov8", "n"): "yolov8n.pt",
    ("yolov8", "m"): "yolov8m.pt",
    ("yolov8", "xl"): "yolov8x.pt",  # "extra large" oficial do YOLOv8 é 'x'
    ("yolov9", "n"): "yolov9t.pt",   # ver nota de rodapé do artigo
    ("yolov9", "m"): "yolov9m.pt",
    ("yolov9", "xl"): "yolov9e.pt",
    ("yolov11", "n"): "yolo11n.pt",
    ("yolov11", "m"): "yolo11m.pt",
    ("yolov11", "xl"): "yolo11x.pt",
}

TRAIN_HPARAMS = dict(
    imgsz=640,
    epochs=100,
    batch=16,
    optimizer="Adam",
    lr0=1e-3,
    cos_lr=True,     # cosine scheduler
    seed=42,
    patience=0,       # sem early stopping, replica fielmente as 100 épocas do artigo
)


def train_one(arch: str, variant: str, data_yaml: str, project_dir: str, device):
    weights = MODEL_MAP[(arch, variant)]
    run_name = f"{arch}_{variant}_base"
    print(f"\n=== Treinando {arch} ({variant}) a partir de {weights} ===")

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        project=project_dir,
        name=run_name,
        device=device,
        **TRAIN_HPARAMS,
    )

    best_weights = Path(project_dir) / run_name / "weights" / "best.pt"
    return str(best_weights), results


def main():
    parser = argparse.ArgumentParser(description="Treinamento das variantes YOLO")
    parser.add_argument("--data", type=str, required=True, help="Caminho para data.yaml")
    parser.add_argument("--project", type=str, default="/home/claude/polyp_yolo_quant/results/runs")
    parser.add_argument("--device", type=str, default="0", help="GPU id ou 'cpu'")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["yolov8", "yolov9", "yolov11"],
        help="Arquiteturas a treinar",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["n", "m", "xl"],
        help="Variantes a treinar",
    )
    args = parser.parse_args()

    manifest = {}
    for arch in args.models:
        for variant in args.variants:
            if (arch, variant) not in MODEL_MAP:
                print(f"[aviso] Combinação não mapeada: {arch}/{variant}, pulando.")
                continue
            weights_path, _ = train_one(arch, variant, args.data, args.project, args.device)
            manifest[f"{arch}_{variant}"] = weights_path

    manifest_path = Path(args.project) / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifesto de pesos treinados salvo em: {manifest_path}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
