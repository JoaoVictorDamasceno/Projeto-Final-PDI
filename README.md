# realtime-polyp-detection-yolo

Reprodução da metodologia do artigo "Otimização de Modelos de Visão
Computacional via Quantização para Detecção de Pólipos em Tempo Real"
(Teixeira et al., ver `docs/references.md`) — variantes do YOLOv8, YOLOv9
e YOLOv11, com e sem quantização FP16 via TensorRT, avaliadas contra o
limiar de 60 FPS.

O projeto roda em **dois modos**, que compartilham todo o código (mesmo split,
mesmo treino, mesma quantização, mesmo benchmark) e diferem só na geração das
caixas do ground truth:

| Modo       | Como ativar  | O que é                                                   |
| ---------- | ------------ | --------------------------------------------------------- |
| `projeto`  | sem flag     | contorno por componente + filtros de ruído (nossa versão) |
| `baseline` | `--baseline` | reprodução fiel ao artigo, sem nenhum filtro              |

Detalhes e justificativa na seção [Modos](#modos-projeto-e-baseline).

## Estrutura

```
realtime-polyp-detection-yolo/
├── configs/
│   └── project_config.yaml      # caminhos e hiperparâmetros de referência
│
├── data/
│   ├── raw/                     # datasets baixados, sem mexer
│   ├── processed/
│   │   ├── projeto/             # dataset do modo projeto (images/, labels/, data.yaml, mode.txt)
│   │   └── baseline/            # dataset do modo baseline (mesma estrutura)
│   └── samples/                 # amostras pequenas pra teste rápido
│
├── src/
│   ├── data_prep/
│   │   ├── mask_to_bbox.py            # conversão de máscara -> bbox (isolada, testável sozinha)
│   │   ├── consolidate_dataset.py     # une os 3 datasets, gera os splits (--baseline)
│   │   ├── validate_dataset.py        # valida o dataset contra a contagem oficial do modo (--baseline)
│   │   ├── verify_mask_filter.py      # confere o filtro de ruído do mask_to_bbox (CVC/ETIS, modo projeto)
│   │   └── scan_multibox_sources.py   # inventário bruto de imagens com mais de 1 pólipo (sem filtro)
│   │
│   ├── training/
│   │   └── train_yolo.py         # fine-tuning das variantes
│   │
│   ├── optimization/
│   │   └── quantize_tensorrt.py  # exportação FP16 via TensorRT
│   │
│   ├── evaluation/
│   │   ├── benchmark.py          # FPS (lendo imagens do disco) + Precisão/Recall/mAP@0.5
│   │   ├── benchmark_mem.py      # FPS (imagens já em memória), sem métricas
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
│   ├── audit/                      # logs da auditoria de bboxes
│   ├── checkpoints/<modo>/         # pesos .pt treinados, engines .engine (TensorRT FP16) e manifestos
│   ├── tables/<modo>/              # CSV de resultados (table1_results.csv, table1_mem.csv)
│   ├── figures/<modo>/             # gráficos gerados
│   └── logs/                       # logs automáticos dos benchmarks (tela + arquivo)
│
├── notebooks/                      # exploração livre
├── docs/
│   ├── decisions_log.md            # registro do que foi tentado/funcionou/não funcionou
│   └── references.md               # citação do artigo base e parâmetros herdados dele
├── pipeline.py                     # roda tudo em sequência (--baseline para o outro modo)
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt --break-system-packages
```

TensorRT não entra no requirements.txt porque depende da GPU/driver de
cada máquina — instalar separado, só na hora de rodar a quantização.

## Modos: projeto e baseline

O artigo descreve a geração das caixas e o protocolo de FPS de forma que deixa
margem para interpretação, e o projeto faz algumas escolhas próprias. Para
medir o efeito de cada uma, os dois modos existem lado a lado. O que muda:

| Etapa                     | `baseline` (fiel ao artigo)                                 | `projeto` (padrão)                                                    |
| ------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| CVC-ClinicDB e ETIS-Larib | **uma** caixa por imagem, com os extremos de toda a máscara | uma caixa por contorno externo, descartando contornos de área < 2 px² |
| HyperKvasir               | todas as caixas do JSON (label `polyp`), como vieram        | descarta a caixa 100% contida em outra da mesma imagem                |
| Total de caixas           | 1879 (1071 + 612 + 196)                                     | 1917 (1063 + 646 + 208)                                               |

O que **não** muda: as 1808 imagens, o split 80/10/10 por imagem com seed 42
(1446/181/181), os hiperparâmetros de treino, a quantização FP16 e o benchmark.
Como os filtros removem caixas e não imagens, a divisão treino/val/teste é
idêntica nos dois modos.

"Extremos da máscara inteira" é a nossa interpretação do artigo (Seção 3.1),
que fala em coordenadas extremas da máscara sem detalhar o caso de várias
lesões. A motivação, as alternativas e a validação estão em
`docs/decisions_log.md` (entrada 007).

Cada modo grava tudo em pastas próprias (`data/processed/<modo>`,
`results/checkpoints/<modo>`, `results/tables/<modo>`, `results/figures/<modo>`),
então um nunca sobrescreve o outro. O `consolidate_dataset.py` grava um marcador
`mode.txt` no dataset, e o `validate_dataset.py` confere se ele bate com a flag.

## Como testar antes de rodar tudo

Antes de rodar o pipeline inteiro em cima do dataset completo, vale testar
a conversão de máscara isolada em poucas imagens:

```bash
python src/data_prep/mask_to_bbox.py --image caminho/img.png --mask caminho/mask.png --out preview.png
```

Abre o `preview.png` gerado e confere se a caixa desenhada realmente está
em volta do pólipo.

## Preparação e auditoria do dataset

```bash
# modo projeto (padrão) -> data/processed/projeto
python src/data_prep/consolidate_dataset.py \
    --hyperkvasir data/raw/HyperKvasir \
    --cvc-clinicdb data/raw/CVC-ClinicDB \
    --etis-larib data/raw/ETIS-LaribPolypDB \
    --out data/processed
python src/data_prep/validate_dataset.py --data-dir data/processed/projeto

# modo baseline -> data/processed/baseline
python src/data_prep/consolidate_dataset.py \
    --hyperkvasir data/raw/HyperKvasir \
    --cvc-clinicdb data/raw/CVC-ClinicDB \
    --etis-larib data/raw/ETIS-LaribPolypDB \
    --out data/processed --baseline
python src/data_prep/validate_dataset.py --data-dir data/processed/baseline --baseline
```

O `--out` é sempre a pasta base (`data/processed`); o modo é acrescentado ao
caminho. O `validate_dataset.py` sai com código 1 se qualquer verificação
falhar, e cada modo tem a sua contagem oficial:

| Modo       | Imagens | Caixas | HyperKvasir | CVC-ClinicDB | ETIS-Larib |
| ---------- | :-----: | :----: | :---------: | :----------: | :--------: |
| `projeto`  |  1808   |  1917  |    1063     |     646      |    208     |
| `baseline` |  1808   |  1879  |    1071     |     612      |    196     |

Para conferir que os dois modos usam as mesmas imagens em cada split:

```bash
for s in train val test; do
  diff <(ls data/processed/projeto/images/$s) <(ls data/processed/baseline/images/$s) && echo "$s: idêntico"
done
```

Outras ferramentas de auditoria (independentes do modo):

```bash
# inventário bruto (sem filtro de área/label) das fontes originais, para
# identificar imagens com mais de 1 pólipo e triá-las visualmente
python src/data_prep/scan_multibox_sources.py \
    --hyperkvasir-dir data/raw/HyperKvasir \
    --cvc-dir data/raw/CVC-ClinicDB \
    --etis-dir data/raw/ETIS-LaribPolypDB \
    --out-logs-dir results/audit \
    --samples-dir data/samples/inventario_bruto

# verificação do filtro de ruído do mask_to_bbox no CVC e no ETIS (modo projeto)
python src/data_prep/verify_mask_filter.py \
    --cvc-dir data/raw/CVC-ClinicDB \
    --etis-dir data/raw/ETIS-LaribPolypDB \
    --samples-dir data/samples/mask_to_bbox_filtrado
```

No modo projeto, o ruído de bboxes é tratado em dois lugares: no
`mask_to_bbox.py`, que descarta contornos de máscara abaixo de
`MIN_CONTOUR_AREA_PX` (CVC-ClinicDB e ETIS-Larib), e no
`consolidate_dataset.py`, que descarta no HyperKvasir a bbox 100% contida em
outra da mesma imagem. O motivo, as evidências e a triagem dos casos de
múltiplos pólipos estão em `docs/decisions_log.md`.

## Rodando o pipeline completo

```bash
# modo projeto
python pipeline.py \
    --hyperkvasir data/raw/HyperKvasir \
    --cvc-clinicdb data/raw/CVC-ClinicDB \
    --etis-larib data/raw/ETIS-LaribPolypDB \
    --device 0

# modo baseline (mesma linha, mais a flag)
python pipeline.py \
    --hyperkvasir data/raw/HyperKvasir \
    --cvc-clinicdb data/raw/CVC-ClinicDB \
    --etis-larib data/raw/ETIS-LaribPolypDB \
    --device 0 --baseline
```

As etapas são: dados, validação, treino, quantização, benchmark,
benchmark_mem e gráficos. O pipeline imprime o modo ativo no início
(`=== MODO: ... ===`) e usa as pastas do modo em todas as etapas.

Cada etapa também roda sozinha (chamando o script direto dentro de `src/`).
Use `--skip` pra pular etapa já feita, por exemplo pra só refazer o
benchmark reaproveitando um treino existente:

```bash
python pipeline.py --skip data validate train quantize ... [--baseline]
```

Os valores aceitos por `--skip` são `data`, `validate`, `train`, `quantize`,
`benchmark`, `benchmark_mem` e `plot`. Rodando um script sozinho, lembre de
passar as pastas do modo certo (por exemplo `--project results/checkpoints/baseline`
no treino), porque os padrões dos scripts não separam por modo.

## Benchmarks de FPS

Existem dois scripts de benchmark, na mesma pasta (`src/evaluation/`). Eles usam
os mesmos pesos `.pt` e engines `.engine` (lidos do `quant_manifest.json`) e só
diferem em **o que entra na medição de FPS**:

| Script             | Entrada do `predict`         | O FPS inclui                                                               |          Métricas           |
| ------------------ | ---------------------------- | -------------------------------------------------------------------------- | :-------------------------: |
| `benchmark.py`     | caminho do arquivo da imagem | leitura e decodificação do JPEG + inferência                               |             sim             |
| `benchmark_mem.py` | array numpy já carregado     | só a inferência (pré e pós-processamento do Ultralytics, sem I/O de disco) | opcional (`--skip-metrics`) |

Por que os dois existem: o artigo define FPS como imagens processadas dividido
pelo tempo total de execução (Seção 3.4), sem dizer se a leitura do disco entra
na conta, e trata os 60 FPS como requisito de **inferência**, separado do
orçamento de 100 ms do pipeline completo. Como o texto é ambíguo, o projeto
reporta as duas interpretações. Detalhes na entrada 006 de
`docs/decisions_log.md`.

```bash
# FPS lendo do disco + Precisão/Recall/mAP@0.5 (tabela completa)
python src/evaluation/benchmark.py \
    --quant-manifest results/checkpoints/<modo>/quant_manifest.json \
    --data data/processed/<modo>/data.yaml \
    --out results/tables/<modo>/table1_results.csv \
    --test-images-dir data/processed/<modo>/images/test

# FPS com imagens em memória (sem métricas: o model.val lê do disco nos dois
# scripts, então as métricas seriam as mesmas)
python src/evaluation/benchmark_mem.py \
    --quant-manifest results/checkpoints/<modo>/quant_manifest.json \
    --data data/processed/<modo>/data.yaml \
    --out results/tables/<modo>/table1_mem.csv \
    --test-images-dir data/processed/<modo>/images/test \
    --skip-metrics
```

Cada modo deve ser avaliado com o seu próprio `data.yaml` e a sua própria pasta
de teste: avaliar um modelo treinado em um modo contra o teste do outro misturaria
ground truths diferentes. Os dois scripts salvam um log da execução em
`results/logs/` (tela + arquivo).

## Notas

- No artigo, o treino ocorreu numa RTX 4090 e o benchmark de inferência
  numa RTX 3070 — pra reproduzir os números da tabela fielmente, rodem
  `benchmark.py` no hardware de inferência-alvo, não no de treino. O hardware
  deste projeto é diferente do usado no artigo: os valores absolutos de FPS (e,
  em menor grau, as métricas) não são diretamente comparáveis. Compare o
  comportamento relativo, como o ganho percentual do FP16 sobre o FP32 e a
  ordem entre os modelos.
- YOLOv9 não tem nomenclatura n/m/xl oficial — usamos YOLOv9t/m/e como
  equivalentes, seguindo a nota de rodapé do artigo.
- Datasets (HyperKvasir, CVC-ClinicDB, ETIS-Larib) não vêm junto do
  repositório — baixem e apontem os caminhos nos argumentos.
- Parâmetros que o artigo não informa e o projeto fixou (`seed=42`,
  `patience=0`, `workspace=4`, `simplify=True`, versões de Ultralytics,
  TensorRT e CUDA) estão listados em `docs/references.md`.
- `configs/project_config.yaml` ainda não está conectado aos scripts, é só
  documentação de referência por enquanto.
- Os logs de `results/audit/*.txt` são versionados; amostras em
  `data/samples/` (incluindo as geradas pelo `scan_multibox_sources.py`),
  previews, checkpoints, engines e `results/logs/` não (ver `.gitignore`).
