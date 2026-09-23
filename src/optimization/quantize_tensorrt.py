"""
Exporta os modelos treinados (FP32) pra engine TensorRT em FP16.
O .pt original fica intacto, guardado separado pra comparação depois.

python quantize_tensorrt.py --manifest results/checkpoints/training_manifest.json

requer: pip install ultralytics tensorrt (tensorrt precisa bater com o
driver CUDA da GPU de inferência)
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

def quantize_model(weights_path: str, imgsz: int, device: str, half: bool = True) -> str:
    model = YOLO(weights_path)
    exported = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=device,
        workspace=4,
        simplify=True,
    )
    return str(exported)

def main():
    parser = argparse.ArgumentParser(description="Quantização FP16 via TensorRT")
    parser.add_argument("--manifest", required=True, help="json do train_yolo.py (modelo -> .pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="id da GPU de inferência")
    parser.add_argument("--out-manifest", default="results/checkpoints/quant_manifest.json")
    args = parser.parse_args()

    with open(args.manifest) as f:
        training_manifest = json.load(f)

    quant_manifest = {}
    for model_key, weights_path in training_manifest.items():
        print(f"\n=== quantizando (FP16) {model_key} ===")
        try:
            engine_path = quantize_model(weights_path, args.imgsz, args.device, half=True)
        except Exception as e:
            print(f"[erro] falha ao quantizar {model_key}: {e}")
            continue

        quant_manifest[model_key] = {
            "base_fp32_pt": weights_path,
            "fp16_engine": engine_path,
        }

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(quant_manifest, f, indent=2)

    print(f"\nmanifesto de quantização salvo em: {out_path}")

if __name__ == "__main__":
    main()