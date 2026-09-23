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

<!-- adicionar entradas abaixo, da mais antiga para a mais recente -->
