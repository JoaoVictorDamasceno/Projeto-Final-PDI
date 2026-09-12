"""
pipeline.py

Encadeia as etapas do projeto na ordem certa:

    1. dados        -> src/data_prep/consolidate_dataset.py
    2. treino        -> src/training/train_yolo.py
    3. quantização   -> src/optimization/quantize_tensorrt.py
    4. benchmark      -> src/evaluation/benchmark.py
    5. gráficos       -> src/evaluation/plot_results.py

Cada script também roda sozinho na sua pasta, isso aqui só encadeia com os
caminhos padrão do projeto. Use --skip pra pular etapa já feita (ex: já
treinou e só quer refazer o benchmark).

python pipeline.py --hyperkvasir X --cvc-clinicdb Y --etis-larib Z --device 0
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
DATA_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "results" / "checkpoints"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo do projeto")
    parser.add_argument("--hyperkvasir", required=True)
    parser.add_argument("--cvc-clinicdb", required=True)
    parser.add_argument("--etis-larib", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["data", "train", "quantize", "benchmark", "plot"],
    )
    args = parser.parse_args()
    py = sys.executable

    if "data" not in args.skip:
        run([py, str(SRC / "data_prep" / "consolidate_dataset.py"),
             "--hyperkvasir", args.hyperkvasir,
             "--cvc-clinicdb", args.cvc_clinicdb,
             "--etis-larib", args.etis_larib,
             "--out", str(DATA_DIR)])

    if "train" not in args.skip:
        run([py, str(SRC / "training" / "train_yolo.py"),
             "--data", str(DATA_DIR / "data.yaml"),
             "--project", str(CKPT_DIR),
             "--device", args.device])

    if "quantize" not in args.skip:
        run([py, str(SRC / "optimization" / "quantize_tensorrt.py"),
             "--manifest", str(CKPT_DIR / "training_manifest.json"),
             "--imgsz", str(args.imgsz),
             "--out-manifest", str(CKPT_DIR / "quant_manifest.json")])

    if "benchmark" not in args.skip:
        run([py, str(SRC / "evaluation" / "benchmark.py"),
             "--quant-manifest", str(CKPT_DIR / "quant_manifest.json"),
             "--data", str(DATA_DIR / "data.yaml"),
             "--out", str(TABLES_DIR / "table1_results.csv"),
             "--imgsz", str(args.imgsz),
             "--test-images-dir", str(DATA_DIR / "images" / "test")])

    if "plot" not in args.skip:
        run([py, str(SRC / "evaluation" / "plot_results.py"),
             "--csv", str(TABLES_DIR / "table1_results.csv"),
             "--outdir", str(FIGURES_DIR)])

    print("\npipeline concluído — resultados em:", ROOT / "results")


if __name__ == "__main__":
    main()
