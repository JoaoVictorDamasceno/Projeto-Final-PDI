"""
benchmark.py

Roda a inferência no hardware de destino pra cada modelo (base FP32 e FP16),
medindo:
  - FPS: imagens processadas / tempo total
  - Precisão, Recall e mAP@0.5 (IoU >= 0.5), via YOLO.val()

Monta a tabela final de comparação (equivalente à Tabela 1 do artigo).

python benchmark.py --quant-manifest results/checkpoints/quant_manifest.json \
    --data data/processed/data.yaml --out results/tables/table1_results.csv \
    --test-images-dir data/processed/images/test
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

FPS_THRESHOLD = 60


def measure_fps(model: YOLO, sample_images, imgsz=640, warmup=10, n_runs=100):
    for _ in range(warmup):
        model.predict(sample_images[0], imgsz=imgsz, verbose=False, device=0)

    start = time.perf_counter()
    for i in range(n_runs):
        img = sample_images[i % len(sample_images)]
        model.predict(img, imgsz=imgsz, verbose=False, device=0)
    elapsed = time.perf_counter() - start

    return n_runs / elapsed


def evaluate_metrics(model: YOLO, data_yaml: str, imgsz=640):
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="test", device=0, verbose=False)
    return float(metrics.box.mp), float(metrics.box.mr), float(metrics.box.map50)


def parse_model_key(model_key: str):
    arch_raw, variant_raw = model_key.split("_")
    arch_display = arch_raw.replace("yolov", "YOLOv")
    variant_display = {"n": "Nano (n)", "m": "Medium (m)", "xl": "Extra Large (xl)"}[variant_raw]
    return arch_display, variant_display


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS + métricas de detecção")
    parser.add_argument("--quant-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--test-images-dir", required=True)
    args = parser.parse_args()

    with open(args.quant_manifest) as f:
        quant_manifest = json.load(f)

    test_images = sorted(str(p) for p in Path(args.test_images_dir).glob("*.*"))
    if not test_images:
        raise SystemExit(f"nenhuma imagem de teste em {args.test_images_dir}")

    rows = []
    for model_key, paths in quant_manifest.items():
        arch_display, variant_display = parse_model_key(model_key)

        for label, weights_path in [("Base (FP32)", paths["base_fp32_pt"]), ("FP16", paths["fp16_engine"])]:
            print(f"\n=== avaliando {arch_display} {variant_display} [{label}] ===")
            model = YOLO(weights_path)

            fps = measure_fps(model, test_images, imgsz=args.imgsz, n_runs=args.n_runs)
            precision, recall, map50 = evaluate_metrics(model, args.data, imgsz=args.imgsz)

            rows.append({
                "Variante": variant_display,
                "Modelo": arch_display,
                "Quantizacao": label,
                "FPS": round(fps, 1),
                "Precisao": round(precision, 3),
                "Recall": round(recall, 3),
                "mAP@0.5": round(map50, 3),
                "Tempo_Real (>=60 FPS)": "Sim" if fps >= FPS_THRESHOLD else "Não",
            })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\nresultados salvos em: {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
