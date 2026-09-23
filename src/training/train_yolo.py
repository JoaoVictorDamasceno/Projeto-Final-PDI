"""
Fine-tuning das variantes nano/medium/xl do YOLOv8, YOLOv9 e YOLOv11 a partir
de pesos pré-treinados em COCO, com os hiperparâmetros usados no artigo:
imgsz 640, 100 épocas, batch 16, Adam (lr 1e-3), cosine scheduler.

Obs: o YOLOv9 não segue a nomenclatura n/m/xl oficialmente. Pra manter a
comparação equivalente, usamos YOLOv9t (~nano), YOLOv9m (~medium) e
YOLOv9e (~extra large), conforme a nota de rodapé do artigo.

requer: pip install ultralytics
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

MODEL_MAP = {
    ("yolov8", "n"): "yolov8n.pt",
    ("yolov8", "m"): "yolov8m.pt",
    ("yolov8", "xl"): "yolov8x.pt",
    ("yolov9", "n"): "yolov9t.pt",
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
    cos_lr=True,
    seed=42,
    patience=0,  # sem early stopping, pra bater as 100 épocas do artigo mesmo
)

def train_one(arch, variant, data_yaml, project_dir, device):
    weights = MODEL_MAP[(arch, variant)]
    run_name = f"{arch}_{variant}_base"
    print(f"\n=== treinando {arch} ({variant}) a partir de {weights} ===")

    model = YOLO(weights)
    model.train(data=data_yaml, project=project_dir, name=run_name, device=device, **TRAIN_HPARAMS)

    # o Ultralytics cria "<nome>2" se a pasta já existe; o trainer sabe o caminho real
    return str(Path(model.trainer.best))

def main():
    parser = argparse.ArgumentParser(description="Treino das variantes YOLO")
    parser.add_argument("--data", required=True, help="caminho pro data.yaml")
    parser.add_argument("--project", default="results/checkpoints")
    parser.add_argument("--device", default="0", help="id da GPU ou 'cpu'")
    parser.add_argument("--models", nargs="+", default=["yolov8", "yolov9", "yolov11"])
    parser.add_argument("--variants", nargs="+", default=["n", "m", "xl"],
                         help="reduza aqui pra treinar menos combinações, ex: --variants m")
    args = parser.parse_args()

    manifest = {}
    for arch in args.models:
        for variant in args.variants:
            if (arch, variant) not in MODEL_MAP:
                print(f"[aviso] combinação não mapeada: {arch}/{variant}, pulando")
                continue
            weights_path = train_one(arch, variant, args.data, args.project, args.device)
            manifest[f"{arch}_{variant}"] = weights_path

    manifest_path = Path(args.project) / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nmanifesto salvo em: {manifest_path}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()