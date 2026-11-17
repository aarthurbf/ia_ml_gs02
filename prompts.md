# Registro de Uso de Inteligência Artificial — Sereno SP
**Global Solution 2026 · FIAP · Módulo de Inteligência Artificial & Machine Learning**

Este documento registra a utilização e interação com ferramentas de Inteligência Artificial Generativa e Assistentes Técnicos durante a concepção, desenvolvimento, refatoração, validação e documentação do módulo de Machine Learning do Sereno SP, em conformidade com as diretrizes da Global Solution 2026.

---

## Registro de Interações — Módulo de IA & ML

### Entrada 01 — Pipeline de Engenharia de Dados e Estruturação do Modelo
* **Data:** 17/11/2026
* **Ferramenta Utilizada:** Claude / IBM Bob (assistente de engenharia de software e ML)
* **Prompt Resumido:**
  > *"Auxiliar na estruturação do pipeline de Machine Learning em Python para previsão de calor extremo na estação INMET A701 (Mirante de Santana), com divisão temporal estrita (treino 2023–2025 e teste 2025–2026), TimeSeriesSplit com 5 dobras, cálculo de limiar P90 sem vazamento e comparação com baselines de persistência e climatológico."*
* **O que foi aproveitado:**
  - Estrutura modular em `sereno_ml/` (`data.py`, `features.py`, `modeling.py`).
  - Lógica de deslocamento temporal diário contínuo por data de calendário (evitando que dias faltantes desloquem posições incorretamente).
  - Scripts de agregação horária e cálculo de métricas de classificação (`metricas()`).
* **O que foi alterado/validado manualmente pela equipe:**
  - Ajuste das faixas físicas meteorológicas aceitas e da regra de anulação de radiação solar diurna (10h–15h).
  - Fixação do critério de exclusão de dias com > 25% de ausência horária (< 18 h válidas) sem qualquer imputação artificial de dados.
  - Definição do limiar de decisão de corte (0,51 / 0,50) focado na meta de negócio (Recall ≥ 0,80 para a classe de calor extremo).
* **Responsável pela revisão técnica:** Arthur Bobadilla Franchi (RM555056), Luan Orlandelli Ramos (RM554747), Jorge Luis (RM554418)

---

### Entrada 02 — Script de Integração e Previsão Operacional/Replay (`predict_publish.py`)
* **Data:** 17/11/2026
* **Ferramenta Utilizada:** Claude / IBM Bob
* **Prompt Resumido:**
  > *"Desenvolver o script CLI `predict_publish.py` para gerar a previsão meteorológica diária a partir de observações recentes ou CSV do INMET, validar faixas físicas, serializar o payload JSON compatível com a entidade PrevisaoCalor da API .NET e publicar via POST /api/previsoes com política de retentativas para erros 5xx/transitórios."*
* **O que foi aproveitado:**
  - Tratamento de argumentos CLI (`--replay`, `--input`, `--csv`, `--dry-run`).
  - Lógica de validação estrita de tipos e faixas numéricas de entrada.
  - Implementação do cliente HTTP com retry exponencial para erros de servidor e recusa imediata para erros de cliente (HTTP 4xx).
* **O que foi alterado/validado manualmente pela equipe:**
  - Alinhamento dos nomes de atributos e convenção do enum `Nivel` (`Perigo`, `Atencao`, `SemAlerta`) com o backend C# .NET.
  - Adição de mensagem explícita no campo `observacao` identificando o modo `--replay` como demonstração retrospectiva com dados históricos, não previsão meteorológica operacional.
* **Responsável pela revisão técnica:** Arthur Bobadilla Franchi (RM555056), Luan Orlandelli Ramos (RM554747), Jorge Luis (RM554418)

---

### Entrada 03 — Curva de Validação, Diagnóstico de Overfitting e Revisão de Documentação
* **Data:** 17/11/2026
* **Ferramenta Utilizada:** Claude / IBM Bob
* **Prompt Resumido:**
  > *"Adicionar a geração da curva de validação variando `max_depth` (de 2 a ilimitado) com TimeSeriesSplit(5) no treino para evidenciar o efeito do overfitting na profundidade das árvores, criar documentação detalhada de dados em `data/README.md`, expandir a suíte de testes unitários em pytest e sintetizar o relatório técnico `RELATORIO_ML.md` nos limites de 3 páginas mantendo todas as tabelas da rubrica."*
* **O que foi aproveitado:**
  - Gráfico da curva de validação (`figures/09_curva_validacao_max_depth.png`) comparando F1 de treino e validação com desvios-padrão entre dobras.
  - Documentação da proveniência dos 7 CSVs semestrais e dicionário de variáveis em `data/README.md`.
  - Testes adicionais em `tests/test_pipeline.py` para robustez da suíte de validação contínua.
* **O que foi alterado/validado manualmente pela equipe:**
  - Conferência cruzada de cada métrica citada no relatório e no roteiro com o arquivo `reports/metricas.json` gerado pelo notebook.
  - Verificação da honestidade científica da comparação com o baseline de persistência.
  - Revisão do enquadramento social voltado aos idosos e trabalhadores de rua de Santana.
* **Responsável pela revisão técnica:** Arthur Bobadilla Franchi (RM555056), Luan Orlandelli Ramos (RM554747), Jorge Luis (RM554418)
