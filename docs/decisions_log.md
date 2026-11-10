# Registro de decisões

Registro do que foi tentado, do que funcionou e do que não funcionou. Cada entrada documenta uma decisão de projeto e a evidência que a sustenta. O código-fonte referencia este arquivo apenas por ponteiro (ex.: `# ver docs/decisions_log.md`).

## Convenção

| Tipo de informação                                | Onde fica                     |
| ------------------------------------------------- | ----------------------------- |
| Contrato do módulo/função (entrada, saída, uso)   | Docstring                     |
| Justificativa curta de valor ou decisão não óbvia | Comentário inline (uma linha) |
| Investigação, alternativas descartadas, histórico | Este arquivo                  |

Cada decisão é uma entrada numerada em ordem cronológica. Entradas antigas não são reescritas: se uma decisão mudar, cria-se uma nova entrada que referencia a anterior e marca a anterior como `Substituída por NNN`.

---

## Modelo de entrada

```markdown
## NNN — Título curto da decisão

**Data:** AAAA-MM-DD
**Status:** Aceita | Substituída por NNN | Descartada
**Scripts/arquivos:** `caminho/do/script.py`

### Problema

O que motivou a decisão. Qual comportamento ou dúvida precisava ser resolvido.

### Método

Como foi investigado. Scripts rodados, parâmetros, amostra analisada.

### Evidências

Números, contagens, casos observados. Preferir dados a impressões.

### Decisão

O que foi decidido e o valor/configuração adotado, com a justificativa.

### Alternativas não adotadas

O que foi considerado e por que foi rejeitado.

### Validação

Como se confirmou que a decisão teve o efeito esperado (ex.: previsto vs. obtido).

### Pendências

O que ficou em aberto, se houver.
```

Seções sem conteúdo podem ser omitidas. `Problema` e `Decisão` são obrigatórias.

---

## Entradas

## 001 — Inventário bruto de bboxes sem filtro (scan_multibox_sources.py)

**Data:** 2026-09-26
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/scan_multibox_sources.py`

### Problema

Precisávamos identificar, nos três datasets crus (HyperKvasir, CVC-ClinicDB,
ETIS-Larib), quais imagens contêm mais de um pólipo, para reduzir o escopo de
checagem visual manual. O `mask_to_bbox.py` e o `consolidate_dataset.py` já
extraem bboxes, mas aplicam filtros de área (e, no caso do HyperKvasir, de
label) antes da extração — o que poderia esconder casos reais de múltiplos
pólipos dentro do próprio filtro, sem chance de revisão.

### Método

Criado o `scan_multibox_sources.py`, que reaproveita o mesmo mecanismo de
extração de bbox já usado no pipeline (JSON bruto para HyperKvasir; máscara
binarizada + `cv2.findContours` + `cv2.boundingRect` para CVC-ClinicDB e
ETIS-Larib), mas sem nenhum filtro de área (`MIN_CONTOUR_AREA_PX` /
`MIN_BOX_AREA_PX`) e sem o filtro de label (`label == "polyp"`) do HyperKvasir.
Rodado nos três datasets completos, gerando um log por dataset com a contagem
de caixas de cada imagem, e uma pasta de amostra visual (`01_original`,
`02_mask`, `03_boxes.png`) para toda imagem com mais de uma caixa.

### Evidências

- HyperKvasir: 1000 imagens no total, 945 com exatamente 1 bbox, 55 com mais
  de 1.
- CVC-ClinicDB: 612 imagens no total, 559 com exatamente 1 bbox, 53 com mais
  de 1, concentradas em sequências de frames consecutivos (ex.: 547–571,
  70–77).
- ETIS-Larib: 196 imagens no total, 190 com exatamente 1 bbox, 6 com mais de
  1, também concentradas em sequência (frames 43–48, todas com 3 caixas).
- A concentração em frames consecutivos de vídeo é evidência a favor de
  pólipos reais nesses casos, já que ruído de máscara (contornos de 1-2px²)
  não tende a se repetir de forma sequencial e consistente.

### Decisão

Manter o scan sem filtro de área ou label, aceitando que parte dos casos de
"mais de 1 bbox" será ruído (esperado e intencional, dado que o objetivo é
reduzir escopo para checagem visual, não decidir automaticamente). A
separação por dataset e a geração de amostras visuais (`samples/`) tornam essa
checagem tratável.

### Alternativas não adotadas

- Aplicar o mesmo filtro de área do `mask_to_bbox.py`/`consolidate_dataset.py`
  diretamente no scan: descartado porque esconderia casos genuínos de
  múltiplos pólipos dentro do próprio filtro, sem chance de revisão manual.

### Validação

Checagem visual de todas as pastas de amostra geradas pelo scan (as imagens
com mais de 1 bbox identificadas nas Evidências), classificando cada caso
como pólipo válido ou possível ruído:

| Dataset      | Casos com >1 bbox | Classificados como ruído | Classificados como válidos |
| ------------ | :---------------: | :----------------------: | :------------------------: |
| HyperKvasir  |        55         |            8             |             47             |
| CVC-ClinicDB |        53         |            23            |             30             |
| ETIS-Larib   |         6         |            0             |             6              |

O outlier de 10 caixas do HyperKvasir (`4604cf0d`) foi classificado como
válido, e não como ruído (ver entrada 003).

### Pendências

Nenhuma quanto à triagem em si — todos os 114 casos de multi-box (55 + 53 + 6)
foram revisados visualmente e classificados (ver Validação). Fica pendente o
tratamento dos 31 casos classificados como ruído (8 HyperKvasir + 23
CVC-ClinicDB), a ser feito em entrada futura referenciando esta.

---

## 002 — Descontinuação do audit_bboxes.py

**Data:** 2026-09-26
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/audit_bboxes.py` (removido)

### Problema

O `audit_bboxes.py` cobria parte da auditoria de bboxes do pipeline, mas
ficou redundante depois da criação do `scan_multibox_sources.py`.

### Decisão

Remover o `audit_bboxes.py`. O `scan_multibox_sources.py` cobre o mesmo
escopo com mais rigor (separação por dataset, amostras visuais, log
completo por imagem), sem deixar lacuna funcional. Verificado que nenhum
outro script do `data_prep/` importa `audit_bboxes`.

### Pendências

Nenhuma.

---

## 003 — Filtro de contenção no HyperKvasir no lugar do filtro de área

**Data:** 2026-10-02
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/consolidate_dataset.py`

### Problema

Na revisão visual (entrada 001), 8 imagens do HyperKvasir foram classificadas
como ruído: cada uma tinha uma caixa espúria. O pipeline antigo tratava isso
com um corte por área (`MIN_BOX_AREA_PX = 200`), que não olhava para a causa do
ruído e podia, em princípio, remover pólipos pequenos reais.

### Método

Substituído o corte por área por um filtro geométrico: uma bbox 100% contida em
outra bbox da mesma imagem é descartada (só a menor). A imagem e as demais
caixas são mantidas. Comparadas as saídas do pipeline antigo e do novo.

### Evidências

- O filtro removeu exatamente 8 caixas, uma em cada imagem de ruído, e nenhuma
  das 47 imagens classificadas como válidas na triagem (inclui a de 10 caixas,
  `4604cf0d`).
- Áreas das 8 caixas removidas: 4, 8, 33, 61, 112, 129, 402 e 1452 px².
- O corte antigo (< 200 px²) pegava só as 6 menores. As caixas de 402 px²
  (`90d30949`) e 1452 px² (`101a484a`) passavam. Isso explica 1925 − 6 = 1919
  no pipeline antigo e 1919 − 2 = 1917 no novo.
- Nenhuma das 6 caixas que o corte antigo removia era pólipo pequeno real.
- Imagens analisadas e no dataset: 1808 e 1808, com split 1446/181/181.

### Decisão

Adotar o filtro de contenção apenas no HyperKvasir. No CVC e no ETIS não se
aplica: a bbox de um componente pode ficar legitimamente dentro da de outro
(ex.: pólipo em C com outro dentro). Removido `MIN_BOX_AREA_PX`.

### Alternativas não adotadas

- Manter o corte por área: deixava 2 caixas de ruído e não tinha base além do
  tamanho.
- Lista fixa de IDs de ruído com remoção manual: não generaliza e esconde o
  critério.

### Validação

Previsto vs. obtido: o cenário previsto (90d30949 com 2 caixas) deu 1917 no
total, com HyperKvasir 1063, CVC 646 e ETIS 208, igual ao obtido. O aviso
`<- VERIFICAR` da 90d30949 era alarme falso: das 3 caixas no JSON, 2 são
pólipos reais e 1 é ruído.

---

## 004 — Redução de MIN_CONTOUR_AREA_PX de 20 para 2 no mask_to_bbox

**Data:** 2026-10-02
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/mask_to_bbox.py`, `src/data_prep/verify_mask_filter.py`

### Problema

No CVC-ClinicDB, 23 imagens foram classificadas como ruído na revisão visual
(entrada 001): contornos minúsculos na máscara geravam caixas extras. O filtro
do `mask_to_bbox.py` usava `MIN_CONTOUR_AREA_PX = 20`; a questão era se esse
valor era o adequado para o ruído observado (contornos de 1–2 px²).

### Método

`MIN_CONTOUR_AREA_PX` alterado de 20 para 2: `mask_to_bbox()` passa a
descartar apenas contornos com área < 2. O `verify_mask_filter.py` roda o
filtro em CVC e ETIS e confere que cada uma das 23 imagens de ruído do CVC
termina com exatamente 1 caixa.

### Evidências

- CVC: 670 contornos brutos − 24 de ruído = 646 caixas.
- ETIS: 208 contornos, nenhum removido.
- Resultado do `verify_mask_filter.py`: 23/23 com 1 box.
- Os 24 contornos removidos estão nas 23 imagens de ruído: 22 tinham 2 caixas
  (1 pólipo + 1 ruído) e 1 tinha 3 caixas (1 pólipo + 2 ruídos), o que dá
  22 + 2 = 24.

### Decisão

Reduzir `MIN_CONTOUR_AREA_PX` de 20 para 2 no `mask_to_bbox`. O valor 2
corresponde ao tamanho do ruído observado na triagem (contornos de 1–2 px²,
entrada 001) e preserva contornos de área ≥ 2 px². Comparados os valores 20 e
2, o resultado foi idêntico (nos dados atuais nenhum contorno tem área entre 2
e 19 px²); optou-se pelo valor mais baixo por ser o mais conservador, já que
descarta menos. O ruído de máscara é tratado na origem, e o consolidate não
aplica filtro adicional ao CVC/ETIS.

### Alternativas não adotadas

- Manter `MIN_CONTOUR_AREA_PX = 20` (valor anterior): o resultado era idêntico
  ao do valor 2, então não havia ganho; o 20 foi descartado por ser menos
  conservador (cortaria contornos pequenos que não são ruído, caso apareçam).
- Filtro de área em pixels da bbox dentro do consolidate (o antigo
  `MIN_BOX_AREA_PX`): ver entrada 003.

---

## 005 — Separação consolidate / validate e contagem oficial do dataset

**Data:** 2026-10-02
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/consolidate_dataset.py`, `src/data_prep/validate_dataset.py`

### Problema

O consolidate carregava valores esperados (`EXPECTED_COUNTS_BY_SOURCE`,
`EXPECTED_TOTAL`, `HYPERKVASIR_NOISE_IDS`) e alertas de divergência. Quem produz
o dataset não deveria se avaliar, e o `validate_dataset.py` estava quebrado
(importava `MIN_BOX_AREA_PX`, que deixou de existir).

### Decisão

- O consolidate só produz o dataset e imprime diagnóstico do que fez (linhas
  `[filtro]`, `[aviso]`, contagens e resumo). Nenhuma comparação com valor
  esperado.
- Todas as verificações contra a contagem oficial ficam no validate: totais e
  contagens por split e por fonte, 1:1 imagem/label, ausência de imagem em dois
  splits, imagens legíveis, formato YOLO, ausência de caixa aninhada no
  HyperKvasir, contagem das 8 imagens de ruído e presença do `data.yaml`.
- Caixas esperadas por fonte (não só total): totais sozinhos podem esconder
  erros que se compensam.
- A contagem por split é só impressa, não verificada: depende do shuffle e não
  há valor independente para comparar.
- Removidos o corte < 200 px² e o argumento `--expected-boxes` do validate.
- Acrescentado um `[aviso]` no consolidate quando uma imagem do CVC/ETIS é
  pulada por não ter máscara (antes era um `continue` silencioso).

### Contagem oficial

| Fonte        | Imagens  |  Caixas  | Origem                             |
| ------------ | :------: | :------: | ---------------------------------- |
| HyperKvasir  |   1000   |   1063   | 1071 no JSON − 8 contidas em outra |
| CVC-ClinicDB |   612    |   646    | 670 contornos − 24 de ruído        |
| ETIS-Larib   |   196    |   208    | nenhum removido                    |
| **Total**    | **1808** | **1917** | split 1446 / 181 / 181             |

Imagens de ruído do HyperKvasir ficam com 9 pólipos reais: 7 imagens com 1 e a
`90d30949` com 2. O dataset tem 109 caixas a mais que 1 por imagem
(63 + 34 + 12), o que pode explicar diferenças em relação ao artigo.

### Validação

Testado com dados sintéticos: o dataset montado com a contagem oficial passa em
todas as verificações, e erros introduzidos de propósito (caixa aninhada,
3 caixas na `90d30949`, caixa extra no CVC, totais) falham.

Rodado nos dados reais (`python src/data_prep/validate_dataset.py --data-dir
data/processed`), o validate terminou com "Dataset válido.", sem nenhuma
verificação em falha.

### Pendências

- Refazer treino, quantização e benchmark: modelos e tabelas anteriores foram
  gerados com o dataset antigo (1919 caixas).
- Se um filtro mudar, o validate falha de propósito até as constantes serem
  atualizadas.

---

## 006 — Benchmark de FPS com imagens em memória (benchmark_mem.py)

**Data:** 2026-11-10
**Status:** Aceita
**Scripts/arquivos:** `src/evaluation/benchmark_mem.py`, `src/evaluation/benchmark.py`, `pipeline.py`

### Problema

O artigo define o FPS como a razão entre o total de imagens processadas e o
tempo total de execução (Seção 3.4), mas não diz se a leitura das imagens do
disco entra nessa conta. Ao mesmo tempo, trata os 60 FPS como requisito de
**inferência**, separado do orçamento de 100 ms do pipeline completo
(pré-processamento, transferência CPU/GPU e renderização). O texto é, portanto,
ambíguo quanto ao que foi cronometrado.

O `benchmark.py` passa o **caminho** de cada imagem ao `predict`, então cada
iteração lê e decodifica um JPEG do disco antes de inferir. Esse custo de I/O
entra no FPS e atua como gargalo, sobretudo nos modelos mais rápidos (nano),
onde a inferência é curta perto da leitura.

### Método

Criado o `benchmark_mem.py`, na mesma pasta do `benchmark.py`. Ele lê todas as
imagens de teste uma única vez (`cv2.imread`, arrays BGR) antes de medir e passa
os arrays ao `predict`, de modo que o FPS não inclui leitura nem decodificação
de disco. Todo o resto é igual ao `benchmark.py`: mesmos pesos `.pt` e engines
`.engine` (lidos do `quant_manifest.json`), `warmup = 10`, `n_runs = 100`
percorrendo as imagens em ciclo, mesmo `imgsz` e mesmo `device`. Opções
próprias: `--skip-metrics` (mede só o FPS) e a saída em `table1_mem.csv`. Os dois
scripts gravam um log da execução em `results/logs/`.

O que o FPS do `benchmark_mem.py` ainda inclui: o pré-processamento do
Ultralytics (redimensionamento e envio à GPU), a inferência e o pós-processamento
(NMS). O que sai: leitura e decodificação do JPEG. Ou seja, é "inferência com a
imagem já em memória", e não o tempo isolado do forward da rede.

### Evidências

- Código: o `benchmark.py` chama `predict` com caminhos de arquivo; o
  `benchmark_mem.py` chama com arrays numpy.
- Artigo: a definição de FPS não menciona leitura de disco, nem a exclui; o
  requisito de 60 FPS é descrito como de inferência (Introdução, Seção 3.4).
- Métricas (Precisão, Recall, mAP@0.5): o `model.val` lê as imagens do disco nos
  dois cenários, então não dependem do benchmark escolhido. Confirmado na
  medição abaixo: as métricas das seis configurações são idênticas nas duas
  tabelas.

Medições (run 1, variantes nano):

| Modelo  | Quantização | FPS (disco) | FPS (memória) | Ganho da memória | Atinge 60 FPS (disco / memória) |
| ------- | ----------- | :---------: | :-----------: | :--------------: | :-----------------------------: |
| YOLOv8  | Base (FP32) |    35.8     |     46.2      |      +29.1%      |            Não / Não            |
| YOLOv8  | FP16        |    53.0     |     84.2      |      +58.9%      |            Não / Sim            |
| YOLOv9  | Base (FP32) |    31.8     |     38.0      |      +19.5%      |            Não / Não            |
| YOLOv9  | FP16        |    50.1     |     74.9      |      +49.5%      |            Não / Sim            |
| YOLOv11 | Base (FP32) |    30.5     |     41.7      |      +36.7%      |            Não / Não            |
| YOLOv11 | FP16        |    51.3     |     74.8      |      +45.8%      |            Não / Sim            |

Leituras da tabela:

- **O resultado do limiar de 60 FPS depende da interpretação.** Lendo as imagens
  do disco, nenhuma das seis configurações atinge 60 FPS. Com as imagens em
  memória, as três configurações FP16 atingem (74.8 a 84.2 FPS), e as três FP32
  continuam abaixo.
- **O custo de leitura é aproximadamente constante por imagem.** A diferença
  entre os tempos por imagem (1/FPS) das duas medições fica entre cerca de 5 e
  9 ms em todas as configurações, o que é consistente com um custo fixo de
  leitura e decodificação do JPEG. Por ser fixo, ele pesa mais nos modelos
  rápidos, e por isso o ganho da memória é maior no FP16 (+46% a +59%) do que no
  FP32 (+20% a +37%).
- **O ganho do FP16 sobre o FP32 aparece menor no benchmark em disco.** No disco,
  o FP16 acelera +48% (YOLOv8), +58% (YOLOv9) e +68% (YOLOv11); em memória,
  +82%, +97% e +79%. O custo fixo de leitura dilui o efeito da quantização.
- **A ordem entre os modelos muda.** No FP16 em memória, o YOLOv8 lidera
  (84.2 FPS), com YOLOv9 e YOLOv11 praticamente empatados (74.9 e 74.8). No FP16
  em disco, os três ficam entre 50.1 e 53.0 FPS.

### Decisão

Manter os dois benchmarks e reportar os dois FPS, sem afirmar qual deles
reproduz o do artigo, já que o texto não permite saber. Ambos são
interpretações adotadas pelo projeto:

- `benchmark.py` (disco): preserva a continuidade com as medições anteriores e é a
  leitura mais literal de "tempo total de execução".
- `benchmark_mem.py` (memória): está mais próximo do requisito de inferência do
  artigo e elimina um gargalo que não existe no uso real, em que os quadros chegam
  da captura de vídeo e não de arquivos JPEG.

A tabela completa (FPS e métricas) sai do `benchmark.py`. O `benchmark_mem.py`
roda com `--skip-metrics`, porque as métricas seriam as mesmas. O limiar de
60 FPS é aplicado a cada coluna separadamente, e uma configuração que passe em
uma e não na outra deve ser reportada como tal.

### Alternativas não adotadas

- Substituir o `benchmark.py` pelo de memória: perderia a comparação com os
  resultados já obtidos e a leitura literal da definição do artigo.
- Cronometrar só o forward da rede (sem pré e pós-processamento): afasta a medição
  do uso real via Ultralytics, em que pré-processamento e NMS fazem parte da
  inferência.
- Medir o pipeline ponta a ponta, com captura e renderização: fora do escopo desta
  etapa; é o tipo de medição que a Fase 7 (`enhancement/`) poderia incluir.

### Validação

Previsto: com os mesmos pesos e a mesma máquina, o FPS em memória deve ser maior
ou igual ao do disco, e as métricas devem coincidir entre os dois scripts.

Obtido (run 1): o FPS em memória foi maior que o do disco nas seis
configurações (+19.5% a +58.9%), e Precisão, Recall e mAP@0.5 coincidiram nas
seis. As duas condições previstas se confirmaram. Os resultados da tabela de
Evidências vêm de uma única execução (`run1`); não há ainda uma medida de
variação entre execuções.

### Pendências

- Repetir os dois benchmarks para ter mais de uma execução e verificar a
  variação do FPS entre rodadas.
- Completar a tabela com a variante medium (a run 1 cobre só a
  nano).
- O `pipeline.py` roda os dois benchmarks por padrão; o `plot_results.py` ainda lê só
  o `table1_results.csv` (FPS lendo do disco).

---

## 007 — Modos baseline (fiel ao artigo) e projeto, com a flag --baseline

**Data:** 2026-11-10
**Status:** Aceita
**Scripts/arquivos:** `src/data_prep/mask_to_bbox.py`, `src/data_prep/consolidate_dataset.py`, `src/data_prep/validate_dataset.py`, `pipeline.py`

### Problema

O objetivo é reproduzir o artigo e propor melhorias. Pelas entradas 003 e 004, o
dataset do projeto diverge do que o artigo descreve: gera uma caixa por contorno
de máscara (o artigo usa as coordenadas extremas de cada máscara), aplica um
filtro de área nas máscaras do CVC/ETIS e um filtro de contenção no HyperKvasir,
e tem 1917 caixas, número que o artigo não informa. Há ainda diferenças de
hardware (treino e inferência em máquinas inferiores às do artigo).

Sem um ponto de comparação fiel, qualquer diferença nos resultados em relação ao
artigo pode vir de várias causas misturadas: hardware, geração das caixas,
filtros, divisão dos dados. Uma melhoria só pode ser afirmada se houver um
baseline limpo para isolar o efeito.

### Método

Criada uma flag única, `--baseline`, que seleciona o modo. Sem a flag, vale o modo
`projeto` (o comportamento existente). A decisão fica em dois pontos do código, e
o modo viaja como parâmetro até lá:

- `mask_to_bbox()` ganhou o parâmetro `whole_mask` (default `False`). Em `True`,
  gera uma única caixa com os extremos de todos os pixels da máscara binarizada,
  sem filtro de área. O limiar de binarização (127) é o mesmo nos dois modos.
- `consolidate_dataset.py` repassa o modo: no baseline usa `whole_mask=True` no
  CVC/ETIS e não aplica o filtro de contenção no HyperKvasir (o filtro por label
  `polyp` permanece, e deve remover nada, o que a contagem confirma).

O `consolidate` grava em `data/processed/<modo>` e escreve um marcador `mode.txt`.
O `validate_dataset.py` recebe a mesma flag, confere o marcador e carrega os
valores esperados do modo. O `pipeline.py` repassa a flag e deriva as pastas de
dados, checkpoints, tabelas e figuras do modo, de modo que um modo nunca
sobrescreve o outro.

### Evidências

Contagem esperada por modo (derivada das entradas 001 a 005; a do baseline deve
ser confirmada na primeira geração):

| Fonte        | Imagens  | Caixas `projeto` | Caixas `baseline` | Origem no baseline                  |
| ------------ | :------: | :--------------: | :---------------: | ----------------------------------- |
| HyperKvasir  |   1000   |       1063       |       1071        | todas as caixas do JSON, sem filtro |
| CVC-ClinicDB |   612    |       646        |        612        | 1 caixa por imagem                  |
| ETIS-Larib   |   196    |       208        |        196        | 1 caixa por imagem                  |
| **Total**    | **1808** |     **1917**     |     **1879**      | diferença de 38 caixas              |

- O split não depende do modo: as mesmas 1808 imagens (todas têm ao menos uma
  caixa) entram nos dois, com a mesma ordem e a mesma seed.
- As 8 imagens de ruído do HyperKvasir voltam a ter a caixa contida no baseline
  (7 imagens com 2 caixas e a `90d30949` com 3).
- No CVC, as 23 imagens classificadas como ruído (entrada 001) passam a ter uma
  caixa esticada até o ruído, e as imagens com mais de um pólipo válido (30 no CVC
  e 6 no ETIS pela triagem) têm as lesões fundidas em uma só caixa. Isso é
  esperado e é o que o artigo provavelmente fez.

### Decisão

Adotar dois modos sobre um único caminho de código, selecionados por `--baseline`:

- **`baseline`**: reprodução fiel ao artigo, sem nenhum filtro. "Extremos da
  máscara inteira" é a **interpretação do projeto** da Seção 3.1, que não detalha
  o caso de várias lesões em uma máscara.
- **`projeto`** (default): contorno por componente, filtro de área mínima no
  CVC/ETIS (entrada 004) e filtro de contenção no HyperKvasir (entrada 003).

Por que uma flag e não código duplicado: a única coisa que muda entre os modos é
o ground truth. Split, treino, quantização e benchmark são idênticos, então um
único caminho de código deixa explícito o que difere (os dois pontos de leitura da
flag) e evita que cópias divirjam sem ninguém perceber.

Por que o default é `projeto`: o modo de risco é o que gera um dataset diferente do
atual. Com `--baseline` explícito, esquecer a flag cai no comportamento já
validado, e não gera um dataset novo em silêncio. O modo ativo é impresso no início
de cada etapa e o nome do modo entra nas pastas de saída.

O baseline serve de ponto de comparação para duas leituras: o efeito isolado dos
filtros e da geração por contorno (baseline vs. projeto) e a validação da
reprodução (baseline vs. artigo, comparando o comportamento relativo, como o ganho
do FP16 e a ordem entre os modelos, e não os valores absolutos de FPS, que
dependem do hardware).

### Alternativas não adotadas

- Dois `data_prep` separados: duplica a leitura dos datasets, o shuffle com seed e a
  escrita no formato YOLO; se uma cópia mudar o split sem querer, os modos deixam
  de usar as mesmas imagens por split.
- Dois projetos `src` inteiros: duplica treino, quantização e benchmark, que não têm
  diferença; qualquer correção teria de ser feita duas vezes e as cópias
  divergiriam.
- Flag `--filter` (sem flag = baseline): o default passaria a ser o modo de risco,
  com erro silencioso se esquecida; além disso, o nome sugeriria "só filtrar
  ruído", quando o baseline também muda o método de geração das caixas.
- Uma chave por filtro (ablação fina): adiada. Se for necessário medir, por exemplo,
  só o filtro de contenção, os dois modos viram presets e a flag se decompõe.

### Validação

Previsto:

- O modo `projeto` regenerado reproduz o dataset anterior: 1917 caixas
  (1063 / 646 / 208) e a mesma divisão treino/val/teste. Serve como teste de
  regressão.
- O modo `baseline` passa no `validate_dataset.py --baseline` com 1879 caixas
  (1071 / 612 / 196), exatamente 1 caixa por imagem no CVC e no ETIS, e as imagens
  de ruído do HyperKvasir com a caixa extra.
- Os dois modos têm listas de arquivos idênticas em cada split (`diff` das pastas
  `images/train`, `val` e `test`).
- Conferência visual no baseline das 23 imagens de ruído do CVC e de algumas
  imagens com mais de um pólipo, para confirmar que a caixa saiu como o artigo
  descreve.

Obtido: todas as verificações previstas foram confirmadas.

- O modo `projeto` regenerado reproduziu o dataset anterior (1917 caixas, com
  1063 / 646 / 208) e passou no `validate_dataset.py`.
- O modo `baseline` passou no `validate_dataset.py --baseline` (1879 caixas, com
  1071 / 612 / 196), incluindo a checagem de exatamente 1 caixa por imagem no CVC
  e no ETIS e a das imagens de ruído do HyperKvasir.
- As listas de arquivos de `train`, `val` e `test` ficaram idênticas entre os dois
  modos.
- A conferência visual do baseline (imagens de ruído do CVC e imagens com mais de
  um pólipo) mostrou a caixa como o artigo descreve.

### Pendências

- Treinar, quantizar e fazer o benchmark do modo `baseline`. No modo `projeto`, a
  pendência da entrada 005 continua valendo se os modelos existentes foram
  treinados com o dataset anterior de 1919 caixas.
- Como as imagens são as mesmas, é possível avaliar um modelo de um modo contra o
  ground truth do outro, de forma deliberada, para separar o efeito do treino do
  efeito do ground truth. Não fazer isso sem querer: cada modelo deve ser avaliado,
  por padrão, com o `data.yaml` do seu próprio modo.
