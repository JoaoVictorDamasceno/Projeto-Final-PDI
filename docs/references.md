# Referências

## Artigo base

TEIXEIRA, Davi de Jesus; OLIVEIRA, Carlos Eduardo Gonçalves de; LIMA, Gustavo
Novack Viana; SANTOS, Rian de Souza; FRANCO, Ricardo Augusto Pereira.
**Otimização de Modelos de Visão Computacional via Quantização para Detecção de
Pólipos em Tempo Real.** In: Simpósio Brasileiro de Computação Aplicada à Saúde
(SBCAS), 26., 2026, Ouro Preto/MG. Anais [...]. Porto Alegre: Sociedade
Brasileira de Computação, 2026. p. 740–751. ISSN 2763-8952.
DOI: https://doi.org/10.5753/sbcas.2026.21502.

Publicado em 01/06/2026. Página do artigo:
https://sol.sbc.org.br/index.php/sbcas/article/view/42805

O PDF não é versionado neste repositório (a página do SOL não informa licença
de reuso).

## O que o projeto herda do artigo

| Item                                                       | Artigo      | Onde está no projeto                                |
| ---------------------------------------------------------- | ----------- | --------------------------------------------------- |
| Fontes de dados e total de 1.808 imagens únicas            | Seção 3.1   | `consolidate_dataset.py`, `validate_dataset.py`     |
| Split 80/10/10 por imagem: 1446 / 181 / 181                | Seção 3.1   | `consolidate_dataset.py` (`SPLIT_RATIOS`)           |
| Pesos pré-treinados no COCO e fine-tuning                  | Seção 3.2   | `train_yolo.py` (`MODEL_MAP`)                       |
| Variantes n, m, xl; YOLOv9 como t, m, e                    | 3.2, nota 1 | `train_yolo.py` (`MODEL_MAP`)                       |
| 640×640, 100 épocas, batch 16, Adam (lr 1e-3), cosine      | Seção 3.3   | `train_yolo.py` (`TRAIN_HPARAMS`)                   |
| Quantização FP16 via TensorRT, modelo base mantido à parte | Seção 3.3   | `quantize_tensorrt.py`                              |
| FPS = imagens processadas / tempo total de execução        | Seção 3.4   | `benchmark.py` (`measure_fps`)                      |
| Limiar de 60 FPS (orçamento de 100 ms, Kader et al. 2026)  | Intro, 3.4  | `benchmark.py`, `plot_results.py` (`FPS_THRESHOLD`) |
| Precisão, Recall e mAP@0.5 com IoU ≥ 0.5                   | Seção 3.4   | `benchmark.py` (`evaluate_metrics`)                 |
| Treino na RTX 4090 e inferência na RTX 3070 (notebook)     | Seção 3.3   | README, seção Notas                                 |

## Onde o projeto diverge ou o artigo não especifica

- **Geração das caixas (CVC-ClinicDB e ETIS-Larib).** O artigo diz que as caixas
  vêm das coordenadas extremas de cada máscara. O projeto gera uma caixa por
  contorno externo da máscara (`mask_to_bbox.py`), com
  `MIN_CONTOUR_AREA_PX = 2`. Ver `docs/decisions_log.md`, entrada 004.
- **Tratamento de ruído.** O artigo não descreve filtragem de caixas. O projeto
  aplica o filtro de contenção no HyperKvasir (entrada 003) e o filtro de
  contornos nas máscaras (entrada 004). O total de 1917 caixas é do projeto; o
  artigo não informa esse número.
- **Parâmetros de treino e exportação não informados.** `seed=42`,
  `patience=0` (sem early stopping), `workspace=4` e `simplify=True` são
  escolhas do projeto. As versões de Ultralytics, TensorRT e CUDA também não
  constam no artigo, que cita apenas o TensorRT Developer Guide 10.0.1.
- **FPS depende do hardware.** O artigo mediu em uma RTX 3070. Os valores só são
  comparáveis se o benchmark rodar em GPU equivalente.

## Referências do artigo usadas diretamente

- Borgli, H. et al. (2020). HyperKvasir, a comprehensive multi-class image and
  video dataset for gastrointestinal endoscopy. _Scientific Data_, 7(1):283.
- Bernal, J. et al. (2015). CVC-ClinicDB.
- Huang, C.-H.; Wu, H.-Y.; Lin, Y.-L. (2024). ETIS-Larib Polyp DB.
- Jocher, G.; Qiu, J.; Chaurasia, A. (2023). Ultralytics YOLO.
- Kader, R. et al. (2026). A novel cloud-based artificial intelligence for
  real-time detection of colorectal neoplasia: a randomized controlled trial
  (EAGLE). _npj Digital Medicine_, 9(84).
