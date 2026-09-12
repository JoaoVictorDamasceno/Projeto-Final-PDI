"""
04_benchmark.py
=================
Executa a inferência no hardware-alvo (ex.: notebook com RTX 3070, conforme
Seção 3.3) para cada modelo (base FP32 e quantizado FP16), medindo:

  - FPS: total de imagens processadas / tempo total de execução
  - Precisão, Recall e mAP@0.5 (IoU >= 0.5), via Ultralytics val()

Reproduz a estrutura da Tabela 1 do artigo, salvando um CSV consolidado.

Uso:
    python 04_benchmark.py \
        --quant-manifest /home/claude/polyp_yolo_quant/results/runs/quant_manifest.json \
        --data /home/claude/polyp_yolo_quant/data/polyp_dataset/data.yaml \
        --out /home/claude/polyp_yolo_quant/results/table1_results.csv
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

FPS_THRESHOLD = 60  # limiar de tempo real adotado no artigo (Seção 1)


def measure_fps(model: YOLO, sample_images, imgsz=640, warmup=10, n_runs=100):
    """Mede FPS = total de imagens processadas / tempo total de execução,
    conforme definido na Seção 3.4."""
    # warmup (evita medir custo de inicialização de kernels/CUDA)
    for _ in range(warmup):
        model.predict(sample_images[0], imgsz=imgsz, verbose=False, device=0)

    start = time.perf_counter()
    n_processed = 0
    for i in range(n_runs):
        img = sample_images[i % len(sample_images)]
        model.predict(img, imgsz=imgsz, verbose=False, device=0)
        n_processed += 1
    elapsed = time.perf_counter() - start

    fps = n_processed / elapsed
    return fps


def evaluate_metrics(model: YOLO, data_yaml: str, imgsz=640):
    """Roda a validação no split de teste e extrai Precisão, Recall e mAP@0.5
    (IoU >= 0.5), conforme Seção 3.4."""
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="test", device=0, verbose=False)
    precision = float(metrics.box.mp)       # mean precision
    recall = float(metrics.box.mr)          # mean recall
    map50 = float(metrics.box.map50)        # mAP@0.5
    return precision, recall, map50


def parse_model_key(model_key: str):
    # ex.: "yolov8_m" -> arquitetura="YOLOv8", variante="Medium (m)"
    arch_raw, variant_raw = model_key.split("_")
    arch_display = arch_raw.replace("yolov", "YOLOv")
    variant_display = {"n": "Nano (n)", "m": "Medium (m)", "xl": "Extra Large (xl)"}[variant_raw]
    return arch_display, variant_display


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS + métricas diagnósticas")
    parser.add_argument("--quant-manifest", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-runs", type=int, default=100,
                         help="Número de inferências para média de FPS")
    parser.add_argument(
        "--test-images-dir", type=str, required=True,
        help="Diretório com imagens do split de teste, usadas para medir FPS",
    )
    args = parser.parse_args()

    with open(args.quant_manifest) as f:
        quant_manifest = json.load(f)

    test_images = sorted(str(p) for p in Path(args.test_images_dir).glob("*.*"))
    if not test_images:
        raise SystemExit(f"Nenhuma imagem de teste encontrada em {args.test_images_dir}")

    rows = []
    for model_key, paths in quant_manifest.items():
        arch_display, variant_display = parse_model_key(model_key)

        for label, weights_path in [
            ("Base (FP32)", paths["base_fp32_pt"]),
            ("FP16", paths["fp16_engine"]),
        ]:
            print(f"\n=== Avaliando {arch_display} {variant_display} [{label}] ===")
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

    print(f"\nResultados salvos em: {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
