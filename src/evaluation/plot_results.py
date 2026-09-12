"""
plot_results.py

Gera dois gráficos a partir do CSV produzido pelo benchmark.py:
  - comparação de FPS (base FP32 vs FP16) por variante/arquitetura
  - trade-off mAP@0.5 x FPS pras configurações FP16

python plot_results.py --csv results/tables/table1_results.csv --outdir results/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VARIANT_ORDER = ["Nano (n)", "Medium (m)", "Extra Large (xl)"]
MODEL_ORDER = ["YOLOv8", "YOLOv9", "YOLOv11"]
FPS_THRESHOLD = 60


def plot_fps_comparison(df: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(len(VARIANT_ORDER), 1, figsize=(8, 10), sharex=True)

    for ax, variant in zip(axes, VARIANT_ORDER):
        sub = df[df["Variante"] == variant]
        x = range(len(MODEL_ORDER))
        width = 0.35

        base_vals, fp16_vals, gains = [], [], []
        for model in MODEL_ORDER:
            row_base = sub[(sub["Modelo"] == model) & (sub["Quantizacao"] == "Base (FP32)")]
            row_fp16 = sub[(sub["Modelo"] == model) & (sub["Quantizacao"] == "FP16")]
            b = row_base["FPS"].values[0] if len(row_base) else 0
            f = row_fp16["FPS"].values[0] if len(row_fp16) else 0
            base_vals.append(b)
            fp16_vals.append(f)
            gains.append(((f - b) / b * 100) if b else 0)

        ax.bar([i - width / 2 for i in x], base_vals, width, label="Base (FP32)", color="white", edgecolor="black")
        ax.bar([i + width / 2 for i in x], fp16_vals, width, label="FP16", color="gray", edgecolor="black", hatch="//")

        for i, (b, f, g) in enumerate(zip(base_vals, fp16_vals, gains)):
            ax.text(i - width / 2, b + 1, f"{b:.1f}", ha="center", fontsize=8)
            ax.text(i + width / 2, f + 1, f"{f:.1f} (+{g:.1f}%)", ha="center", fontsize=8)

        ax.axhline(FPS_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel("FPS")
        ax.set_title(variant, loc="right", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylim(0, max(fp16_vals + base_vals + [FPS_THRESHOLD]) * 1.25)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("FPS por arquitetura/variante — base FP32 vs FP16")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"gráfico de FPS salvo em: {outpath}")


def plot_tradeoff(df: pd.DataFrame, outpath: Path):
    fp16_df = df[df["Quantizacao"] == "FP16"]

    markers = {"YOLOv8": "o", "YOLOv9": "s", "YOLOv11": "D"}
    facecolors = {"Nano (n)": "white", "Medium (m)": "gray", "Extra Large (xl)": "black"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in fp16_df.iterrows():
        ax.scatter(
            row["FPS"], row["mAP@0.5"],
            marker=markers.get(row["Modelo"], "o"),
            s=90,
            facecolors=facecolors.get(row["Variante"], "gray"),
            edgecolors="black",
            linewidths=1.2,
        )

    ax.axvline(FPS_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label="limiar 60 FPS")
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP@0.5 x FPS — variantes FP16")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"gráfico de trade-off salvo em: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Gráficos comparativos dos resultados")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_fps_comparison(df, outdir / "fps_comparacao.png")
    plot_tradeoff(df, outdir / "tradeoff_map_fps.png")


if __name__ == "__main__":
    main()
