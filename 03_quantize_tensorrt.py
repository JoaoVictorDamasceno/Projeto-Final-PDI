"""
03_quantize_tensorrt.py
=========================
Aplica a quantização FP16 via NVIDIA TensorRT aos modelos treinados
(base FP32), conforme Seção 3.3 do artigo.

O modelo base FP32 é mantido intacto para comparação posterior (Tabela 1).
A engine TensorRT FP16 é exportada para o mesmo diretório dos pesos base.

Uso:
    python 03_quantize_tensorrt.py \
        --manifest /home/claude/polyp_yolo_quant/results/runs/training_manifest.json \
        --imgsz 640

Requer: pip install ultralytics tensorrt --break-system-packages
(TensorRT deve estar disponível no hardware alvo de inferência, ex.: RTX 3070)
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def quantize_model(weights_path: str, imgsz: int, half: bool = True) -> str:
    """Exporta um modelo .pt para engine TensorRT (.engine), com ou sem FP16."""
    model = YOLO(weights_path)
    exported_path = model.export(
        format="engine",   # TensorRT
        imgsz=imgsz,
        half=half,          # True => FP16 ; False => mantém FP32
        device=0,
        workspace=4,         # GB de workspace para o builder do TensorRT
        simplify=True,
    )
    return str(exported_path)


def main():
    parser = argparse.ArgumentParser(description="Quantização FP16 via TensorRT")
    parser.add_argument("--manifest", type=str, required=True,
                         help="JSON gerado por 02_train_models.py mapeando modelo -> pesos .pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-manifest", type=str,
                         default="/home/claude/polyp_yolo_quant/results/runs/quant_manifest.json")
    args = parser.parse_args()

    with open(args.manifest) as f:
        training_manifest = json.load(f)

    quant_manifest = {}
    for model_key, weights_path in training_manifest.items():
        print(f"\n=== Quantizando (FP16) {model_key} ===")
        try:
            engine_fp16 = quantize_model(weights_path, args.imgsz, half=True)
        except Exception as e:
            print(f"[erro] Falha ao quantizar {model_key}: {e}")
            continue

        quant_manifest[model_key] = {
            "base_fp32_pt": weights_path,
            "fp16_engine": engine_fp16,
        }

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as f:
        json.dump(quant_manifest, f, indent=2)

    print(f"\nManifesto de quantização salvo em: {args.out_manifest}")


if __name__ == "__main__":
    main()
