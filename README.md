# realtime-polyp-detection-yolo

Reprodução da metodologia do artigo "Otimização de Modelos de Visão
Computacional via Quantização para Detecção de Pólipos em Tempo Real"
(Teixeira et al.) — variantes do YOLOv8, YOLOv9 e YOLOv11, com e sem
quantização FP16 via TensorRT, avaliadas contra o limiar de 60 FPS.

## Estrutura

```
realtime-polyp-detection-yolo/
├── configs/
│   └── project_config.yaml      # caminhos e hiperparâmetros de referência
│
├── data/
│   ├── raw/                     # datasets baixados, sem mexer
│   ├── processed/                # dataset consolidado em formato YOLO
│   └── samples/                  # amostras pequenas pra teste rápido
│
├── src/
│   ├── data_prep/
│   │   ├── mask_to_bbox.py       # conversão de máscara -> bbox (isolada, testável sozinha)
│   │   └── consolidate_dataset.py # une os 3 datasets, gera os splits
│   │
│   ├── training/
│   │   └── train_yolo.py         # fine-tuning das variantes
│   │
│   ├── optimization/
│   │   └── quantize_tensorrt.py  # exportação FP16 via TensorRT
│   │
│   ├── evaluation/
│   │   ├── benchmark.py          # FPS + Precisão/Recall/mAP@0.5
│   │   └── plot_results.py       # gráficos de FPS e trade-off mAP x FPS
│   │
│   ├── enhancement/               # Fase 7 — ainda não implementado
│   │   ├── preprocessing.py
│   │   └── postprocessing.py
│   │
│   └── inference/
│       └── video_demo.py          # demonstração final em vídeo, ainda não implementado
│
├── results/
│   ├── checkpoints/                # pesos .pt treinados
│   ├── engines/                    # engines .engine (TensorRT FP16)
│   ├── tables/                     # CSV de resultados
│   └── figures/                    # gráficos gerados
│
├── notebooks/                      # exploração livre
├── docs/
│   └── decisions_log.md            # registro do que foi tentado/funcionou/não funcionou
├── pipeline.py                     # roda tudo em sequência
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt --break-system-packages
```

TensorRT não entra no requirements.txt porque depende da GPU/driver de
cada máquina — instalar separado, só na hora de rodar a quantização.

## Como testar antes de rodar tudo

Antes de rodar o pipeline inteiro em cima do dataset completo, vale testar
a conversão de máscara isolada em poucas imagens:

```bash
python src/data_prep/mask_to_bbox.py --image caminho/img.png --mask caminho/mask.png --out preview.png
```

Abre o `preview.png` gerado e confere se a caixa desenhada realmente está
em volta do pólipo.

## Rodando o pipeline completo

```bash
python pipeline.py \
    --hyperkvasir data/raw/hyperkvasir \
    --cvc-clinicdb data/raw/cvc_clinicdb \
    --etis-larib data/raw/etis_larib \
    --device 0
```

Cada etapa também roda sozinha (chamando o script direto dentro de `src/`).
Use `--skip` pra pular etapa já feita, por exemplo pra só refazer o
benchmark reaproveitando um treino existente:

```bash
python pipeline.py --skip data train ...
```

## Notas

- No artigo, o treino ocorreu numa RTX 4090 e o benchmark de inferência
  numa RTX 3070 — pra reproduzir os números da tabela fielmente, rodem
  `benchmark.py` no hardware de inferência-alvo, não no de treino.
- YOLOv9 não tem nomenclatura n/m/xl oficial — usamos YOLOv9t/m/e como
  equivalentes, seguindo a nota de rodapé do artigo.
- Datasets (HyperKvasir, CVC-ClinicDB, ETIS-Larib) não vêm junto do
  repositório — baixem e apontem os caminhos nos argumentos.
- `configs/project_config.yaml` ainda não está conectado aos scripts, é só
  documentação de referência por enquanto.
