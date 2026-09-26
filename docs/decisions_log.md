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
  de 1 (inclui um outlier de 10 caixas em uma única imagem).
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
| HyperKvasir  |        55         | 8 (inclui outlier de 10) |             47             |
| CVC-ClinicDB |        53         |            23            |             30             |
| ETIS-Larib   |         6         |            0             |             6              |

O outlier de 10 caixas (HyperKvasir) está entre os 8 casos classificados
como ruído, confirmando a suspeita levantada na fase de Evidências.

### Pendências

Nenhuma quanto à triagem em si — todos os 114 casos de multi-box (55 + 53 + 6)
foram revisados visualmente e classificados (ver Validação). Fica pendente o
tratamento dos 31 casos classificados como ruído (8 HyperKvasir + 23
CVC-ClinicDB), a ser feito em entrada futura referenciando esta.

---

## 002 — Descontinuação do audit_bboxes.py

**Data:** 2026-09-26
**Status:** Substituída por 001
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
