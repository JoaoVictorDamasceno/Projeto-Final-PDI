"""
Roda a inferência no hardware de destino pra cada modelo (base FP32 e FP16),
medindo:
  - FPS: imagens processadas / tempo total
  - Precisão, Recall e mAP@0.5 (IoU >= 0.5), via YOLO.val()

Monta a tabela final de comparação (equivalente à Tabela 1 do artigo).

python src/evaluation/benchmark.py --quant-manifest results/checkpoints/quant_manifest.json \
    --data data/processed/data.yaml --out results/tables/table1_results.csv \
    --test-images-dir data/processed/images/test
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ---- log automático em results/logs/ (tela + arquivo) ----------------------
# Instalado antes de importar o ultralytics para que os avisos dele também
# entrem no arquivo. A pasta pode ser trocada com --log-dir.
ROOT = Path(__file__).resolve().parents[2]  # raiz do projeto (/workspace)


class _Tee:
    """Replica tudo que é escrito na tela também no arquivo de log."""

    def __init__(self, stream, logfile):
        self._stream, self._logfile = stream, logfile

    def write(self, text):
        self._stream.write(text)
        self._logfile.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def __getattr__(self, name):  # isatty(), fileno(), encoding...
        return getattr(self._stream, name)


def _setup_log(prefix):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--log-dir", default=str(ROOT / "results" / "logs"))
    known, _ = pre.parse_known_args()
    log_dir = Path(known.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logfile = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)
    return path


LOG_PATH = _setup_log("benchmark")

import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402

FPS_THRESHOLD = 60

def measure_fps(model: YOLO, sample_images, imgsz=640, warmup=10, n_runs=100, device="0"):
    for _ in range(warmup):
        model.predict(sample_images[0], imgsz=imgsz, verbose=False, device=device)

    start = time.perf_counter()
    for i in range(n_runs):
        img = sample_images[i % len(sample_images)]
        model.predict(img, imgsz=imgsz, verbose=False, device=device)
    elapsed = time.perf_counter() - start

    return n_runs / elapsed

def evaluate_metrics(model: YOLO, data_yaml: str, imgsz=640, device="0"):
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="test", device=device, verbose=False)
    return float(metrics.box.mp), float(metrics.box.mr), float(metrics.box.map50)

def parse_model_key(model_key: str):
    arch_raw, variant_raw = model_key.split("_")
    arch_display = arch_raw.replace("yolov", "YOLOv")
    variant_display = {"n": "Nano (n)", "m": "Medium (m)"}[variant_raw]
    return arch_display, variant_display

def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS + métricas de detecção")
    parser.add_argument("--quant-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--device", default="0", help="id da GPU de inferência")
    parser.add_argument("--test-images-dir", required=True)
    parser.add_argument("--log-dir", default=str(ROOT / "results" / "logs"),
                        help="pasta onde o log desta execução é salvo")
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

            fps = measure_fps(model, test_images, imgsz=args.imgsz, n_runs=args.n_runs, device=args.device)
            precision, recall, map50 = evaluate_metrics(model, args.data, imgsz=args.imgsz, device=args.device)

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
    print(f"\nlog desta execução: {LOG_PATH}")

if __name__ == "__main__":
    main()