"""
Encadeia as etapas do projeto na ordem certa:

    1. dados          -> src/data_prep/consolidate_dataset.py
    2. validação      -> src/data_prep/validate_dataset.py
    3. treino         -> src/training/train_yolo.py
    4. quantização    -> src/optimization/quantize_tensorrt.py
    5. benchmark      -> src/evaluation/benchmark.py       (FPS lendo imagens do disco + métricas)
    6. benchmark_mem  -> src/evaluation/benchmark_mem.py   (FPS com imagens em memória, sem métricas)
    7. gráficos       -> src/evaluation/plot_results.py

Modos (uma única flag):

    sem flag     -> modo "projeto": caixas por contorno + filtros (1917 caixas)
    --baseline   -> modo "baseline": reprodução fiel ao artigo, sem filtros (1879 caixas)

Tudo o que o modo produz fica em uma pasta própria, para um modo nunca
sobrescrever o outro:

    data/processed/<modo>/
    results/checkpoints/<modo>/
    results/tables/<modo>/
    results/figures/<modo>/

Cada script também roda sozinho na sua pasta, isso aqui só encadeia com os
caminhos padrão do projeto. Use --skip pra pular etapa já feita (ex: já
treinou e só quer refazer o benchmark).

A validação confere o dataset consolidado contra a contagem oficial do modo e
sai com código 1 se algo divergir, o que interrompe o pipeline antes do treino.

python pipeline.py \
    --hyperkvasir data/raw/HyperKvasir \
    --cvc-clinicdb data/raw/CVC-ClinicDB \
    --etis-larib data/raw/ETIS-LaribPolypDB \
    --device 0 [--baseline]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
DATA_BASE = ROOT / "data" / "processed"        # o consolidate acrescenta <modo> a este caminho
CKPT_BASE = ROOT / "results" / "checkpoints"
TABLES_BASE = ROOT / "results" / "tables"
FIGURES_BASE = ROOT / "results" / "figures"

STEPS = ["data", "validate", "train", "quantize", "benchmark", "benchmark_mem", "plot"]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo do projeto")
    parser.add_argument("--hyperkvasir", required=True)
    parser.add_argument("--cvc-clinicdb", required=True)
    parser.add_argument("--etis-larib", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="+", help="ex: --models yolov8 yolov11 (padrão: todos)")
    parser.add_argument("--variants", nargs="+", help="ex: --variants m (padrão: n m xl)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline", action="store_true",
                        help="reprodução fiel ao artigo: caixa única pela máscara inteira "
                             "(CVC/ETIS) e HyperKvasir sem filtro de contenção")
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    args = parser.parse_args()
    py = sys.executable

    mode = "baseline" if args.baseline else "projeto"
    data_dir = DATA_BASE / mode
    ckpt_dir = CKPT_BASE / mode
    tables_dir = TABLES_BASE / mode
    figures_dir = FIGURES_BASE / mode
    print(f"=== MODO: {mode} ===")

    if "data" not in args.skip:
        cmd = [py, str(SRC / "data_prep" / "consolidate_dataset.py"),
               "--hyperkvasir", args.hyperkvasir,
               "--cvc-clinicdb", args.cvc_clinicdb,
               "--etis-larib", args.etis_larib,
               "--out", str(DATA_BASE)]  # o consolidate grava em <out>/<modo>
        if args.baseline:
            cmd.append("--baseline")
        run(cmd)

    if "validate" not in args.skip:
        cmd = [py, str(SRC / "data_prep" / "validate_dataset.py"),
               "--data-dir", str(data_dir)]
        if args.baseline:
            cmd.append("--baseline")
        run(cmd)

    if "train" not in args.skip:
        train_cmd = [py, str(SRC / "training" / "train_yolo.py"),
                     "--data", str(data_dir / "data.yaml"),
                     "--project", str(ckpt_dir),
                     "--device", args.device]
        if args.models:
            train_cmd += ["--models", *args.models]
        if args.variants:
            train_cmd += ["--variants", *args.variants]
        run(train_cmd)

    if "quantize" not in args.skip:
        run([py, str(SRC / "optimization" / "quantize_tensorrt.py"),
             "--manifest", str(ckpt_dir / "training_manifest.json"),
             "--imgsz", str(args.imgsz),
             "--device", args.device,
             "--out-manifest", str(ckpt_dir / "quant_manifest.json")])

    if "benchmark" not in args.skip:
        run([py, str(SRC / "evaluation" / "benchmark.py"),
             "--quant-manifest", str(ckpt_dir / "quant_manifest.json"),
             "--data", str(data_dir / "data.yaml"),
             "--out", str(tables_dir / "table1_results.csv"),
             "--imgsz", str(args.imgsz),
             "--device", args.device,
             "--test-images-dir", str(data_dir / "images" / "test")])

    if "benchmark_mem" not in args.skip:
        # só FPS: as métricas já saem do benchmark.py (model.val lê do disco nos dois)
        run([py, str(SRC / "evaluation" / "benchmark_mem.py"),
             "--quant-manifest", str(ckpt_dir / "quant_manifest.json"),
             "--data", str(data_dir / "data.yaml"),
             "--out", str(tables_dir / "table1_mem.csv"),
             "--imgsz", str(args.imgsz),
             "--device", args.device,
             "--test-images-dir", str(data_dir / "images" / "test"),
             "--skip-metrics"])

    if "plot" not in args.skip:
        run([py, str(SRC / "evaluation" / "plot_results.py"),
             "--csv", str(tables_dir / "table1_results.csv"),
             "--outdir", str(figures_dir)])

    print(f"\npipeline ({mode}) concluído — resultados em:", ROOT / "results")


if __name__ == "__main__":
    main()