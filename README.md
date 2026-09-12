# Otimização de Modelos de Visão Computacional via Quantização para Detecção de Pólipos em Tempo Real

Implementação em Python da metodologia descrita no artigo (Teixeira et al.),
que avalia variantes **nano (n)**, **medium (m)** e **extra large (xl)** dos
modelos **YOLOv8**, **YOLOv9** e **YOLOv11** — com e sem quantização **FP16**
via **NVIDIA TensorRT** — para detecção de pólipos colorretais em tempo real
(limiar de **60 FPS**).

## Estrutura do projeto

```
polyp_yolo_quant/
├── requirements.txt
├── run_pipeline.py              # orquestrador do pipeline completo (Figura 2 do artigo)
├── scripts/
│   ├── 01_prepare_dataset.py    # Etapa 1: consolidação do dataset (HyperKvasir+CVC-ClinicDB+ETIS-Larib)
│   ├── 02_train_models.py       # Etapa 2: fine-tuning das variantes YOLO (n/m/xl)
│   ├── 03_quantize_tensorrt.py  # Etapa 3: quantização FP16 via TensorRT
│   ├── 04_benchmark.py          # Etapa 4: FPS + Precisão/Recall/mAP@0.5 (Tabela 1)
│   └── 05_plot_results.py       # Figuras 3 (FPS) e 4 (trade-off mAP x FPS)
├── data/                        # dataset consolidado (gerado pela Etapa 1)
└── results/                     # pesos treinados, engines TensorRT, CSV e figuras
```

## Instalação

```bash
pip install -r requirements.txt --break-system-packages
# TensorRT precisa ser instalado separadamente, compatível com a GPU/driver
# CUDA do hardware de inferência (ex.: RTX 3070, como no artigo).
```

## Como o pipeline replica o artigo

| Etapa do artigo (Seção) | Script | O que faz |
|---|---|---|
| 3.1 Conjunto de Dados | `01_prepare_dataset.py` | Une HyperKvasir (boxes nativas), CVC-ClinicDB e ETIS-Larib (boxes derivadas de máscaras via coordenadas extremas), particiona 80/10/10 |
| 3.2 Modelos de Detecção | `02_train_models.py` | Fine-tuning das variantes n/m/xl de YOLOv8, YOLOv9 (t/m/e) e YOLOv11 a partir de pesos COCO |
| 3.3 Treinamento | `02_train_models.py` | imgsz=640, 100 épocas, batch=16, Adam (lr=1e-3), cosine scheduler |
| 3.3 Quantização | `03_quantize_tensorrt.py` | Exporta cada modelo treinado para engine TensorRT FP16, mantendo o `.pt` FP32 original |
| 3.4 Métricas | `04_benchmark.py` | Mede FPS (imagens processadas / tempo total) e Precisão/Recall/mAP@0.5 (IoU≥0.5) no split de teste, reproduzindo a Tabela 1 |
| Figuras 3 e 4 | `05_plot_results.py` | Gráfico de barras de FPS (Base vs FP16) e dispersão mAP@0.5 × FPS |

## Uso rápido (pipeline completo)

```bash
python run_pipeline.py \
    --hyperkvasir /caminho/para/hyperkvasir \
    --cvc-clinicdb /caminho/para/cvc_clinicdb \
    --etis-larib /caminho/para/etis_larib \
    --device 0
```

Cada etapa também roda isoladamente, por exemplo, apenas quantização e benchmark
reaproveitando um treino já feito:

```bash
python run_pipeline.py --skip prepare train ...
```

## Notas importantes

- **Hardware**: no artigo, o treino ocorreu em uma RTX 4090 e a *inferência/benchmark*
  em uma RTX 3070 (notebook). Para reproduzir os números da Tabela 1 fielmente,
  rode `04_benchmark.py` no hardware de inferência-alvo, não no de treino.
- **YOLOv9**: não possui nomenclatura n/m/xl oficial; o script usa
  `YOLOv9t` (~nano), `YOLOv9m` (~medium) e `YOLOv9e` (~extra large), conforme a
  nota de rodapé do artigo.
- **Datasets**: HyperKvasir, CVC-ClinicDB e ETIS-Larib são públicos, mas não
  estão embutidos neste repositório — baixe-os e aponte os caminhos via `--hyperkvasir`,
  `--cvc-clinicdb`, `--etis-larib`.
- Os scripts assumem `ultralytics` (que já expõe YOLOv8, YOLOv9 e YOLOv11 sob uma
  API unificada) e exportação `format="engine"` para gerar as engines TensorRT.
