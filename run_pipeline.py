"""
run_pipeline.py
=================
Orquestra o pipeline completo replicando a metodologia da Figura 2 do artigo:

    1. CONJUNTO DE DADOS  -> 01_prepare_dataset.py
    2. MODELOS (treino)    -> 02_train_models.py
    3. OTIMIZAÇÃO (FP16)   -> 03_quantize_tensorrt.py
    4. INFERÊNCIA/MÉTRICAS -> 04_benchmark.py
    (extra) FIGURAS        -> 05_plot_results.py

Cada etapa também pode ser executada isoladamente chamando o script
correspondente em scripts/. Este orquestrador apenas encadeia as chamadas
com os caminhos padrão do projeto.

Uso (exemplo):
    python run_pipeline.py \
        --hyperkvasir /data/hyperkvasir \
        --cvc-clinicdb /data/cvc_clinicdb \
        --etis-larib /data/etis_larib \
        --device 0
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data" / "polyp_dataset"
RUNS_DIR = ROOT / "results" / "runs"
RESULTS_CSV = ROOT / "results" / "table1_results.csv"


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo do artigo")
    parser.add_argument("--hyperkvasir", type=str, required=True)
    parser.add_argument("--cvc-clinicdb", type=str, required=True)
    parser.add_argument("--etis-larib", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["prepare", "train", "quantize", "benchmark", "plot"],
        help="Etapas a pular (útil para re-executar apenas parte do pipeline)",
    )
    args = parser.parse_args()
    py = sys.executable

    if "prepare" not in args.skip:
        run([py, str(SCRIPTS / "01_prepare_dataset.py"),
             "--hyperkvasir", args.hyperkvasir,
             "--cvc-clinicdb", args.cvc_clinicdb,
             "--etis-larib", args.etis_larib,
             "--out", str(DATA_DIR)])

    if "train" not in args.skip:
        run([py, str(SCRIPTS / "02_train_models.py"),
             "--data", str(DATA_DIR / "data.yaml"),
             "--project", str(RUNS_DIR),
             "--device", args.device])

    if "quantize" not in args.skip:
        run([py, str(SCRIPTS / "03_quantize_tensorrt.py"),
             "--manifest", str(RUNS_DIR / "training_manifest.json"),
             "--imgsz", str(args.imgsz),
             "--out-manifest", str(RUNS_DIR / "quant_manifest.json")])

    if "benchmark" not in args.skip:
        run([py, str(SCRIPTS / "04_benchmark.py"),
             "--quant-manifest", str(RUNS_DIR / "quant_manifest.json"),
             "--data", str(DATA_DIR / "data.yaml"),
             "--out", str(RESULTS_CSV),
             "--imgsz", str(args.imgsz),
             "--test-images-dir", str(DATA_DIR / "images" / "test")])

    if "plot" not in args.skip:
        run([py, str(SCRIPTS / "05_plot_results.py"),
             "--csv", str(RESULTS_CSV),
             "--outdir", str(ROOT / "results")])

    print("\n✅ Pipeline concluído. Resultados em:", ROOT / "results")


if __name__ == "__main__":
    main()
