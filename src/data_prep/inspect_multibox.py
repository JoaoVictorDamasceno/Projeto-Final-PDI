"""
Inspeção visual de labels com múltiplas bboxes.

Para cada label informado, imprime dimensões e área em pixels de cada bbox
(calculadas com o tamanho real da imagem) e salva a imagem original com as
boxes desenhadas e numeradas em --out. Boxes com área < 1000px² são
sinalizadas no console.

Uso:
    python inspect_multibox.py \
        --labels data/processed/labels/train/cvcclinicdb_547.txt \
                 data/processed/labels/train/etislarib_44.txt \
        --images-dir data/processed/images/train \
        --out data/processed/_inspecao_multibox
"""

import argparse
from pathlib import Path

import cv2


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_path in args.labels:
        label_path = Path(label_path)
        img_path = find_image_for_label(label_path, images_dir)
        if img_path is None:
            print(f"[aviso] imagem não encontrada pra {label_path.name}, pulando")
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        print(f"\n{label_path.name} ({len(lines)} bbox(es), imagem {w}x{h}):")

        for i, line in enumerate(lines):
            _, cx, cy, bw, bh = map(float, line.split())
            box_w_px = bw * w
            box_h_px = bh * h
            area_px = box_w_px * box_h_px

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            flag = "  <- MUITO PEQUENA, suspeita de ruído" if area_px < 1000 else ""
            print(f"  box {i}: {box_w_px:.0f}x{box_h_px:.0f}px, área={area_px:.0f}px²{flag}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{i}:{area_px:.0f}px2", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = out_dir / f"{label_path.stem}_boxes.png"
        cv2.imwrite(str(out_path), img)
        print(f"  salvo em: {out_path}")


if __name__ == "__main__":
    main()