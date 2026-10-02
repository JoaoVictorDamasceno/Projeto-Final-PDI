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
