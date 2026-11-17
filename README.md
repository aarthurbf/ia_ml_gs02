# Sereno SP — Módulo de IA & Machine Learning
Previsão, na véspera, de **dia de calor extremo** em São Paulo (estação INMET **A701 – Mirante de Santana**) com
**Random Forest** (scikit-learn). Global Solution 2026 · FIAP.

| Integrantes | RM |
|---|---|
| Arthur Bobadilla Franchi | RM555056 |
| Luan Orlandelli Ramos | RM554747 |
| Jorge Luis | RM554418 |

| Resultado no teste (17/11/2025 → 16/11/2026) | Valor |
|---|---|
| Limiar de calor extremo (P90 do treino) | 32,9 °C |
| Recall / Precision / F1 (classe 1) | 0,810 / 0,607 / 0,694 |
| ROC‑AUC / PR‑AUC | 0,965 / 0,775 |
| Baseline persistência (F1) | 0,714 |

Detalhes e leitura crítica: [`RELATORIO_ML.md`](RELATORIO_ML.md).

## Estrutura
```text
ml/
├── notebook_sereno_calor.ipynb   # pipeline completo, executado, com saídas visíveis
├── RELATORIO_ML.md               # relatório acadêmico
├── ROTEIRO_VIDEO.md              # roteiro do vídeo (5 min)
├── README.md
├── requirements.txt              # versões fixadas
├── predict_publish.py            # previsão + POST /api/previsoes (replay e operacional)
├── sereno_ml/                    # código reutilizável
│   ├── data.py                   #   leitura, limpeza, agregação diária
│   ├── features.py               #   atributos e variável-alvo
│   └── modeling.py               #   divisão temporal, baselines, métricas
├── tests/test_pipeline.py        # 20 testes (pytest)
├── tools/                        # notebook_src.py + nbtools.py (gera/executa o .ipynb)
├── data/raw/                     # 7 CSVs semestrais do INMET A701
├── data/processed/diario_a701.csv
├── models/sereno_rf.joblib       # modelo + metadados (limiares, atributos, versão)
├── figures/                      # gráficos do relatório (PNG, 150 dpi)
└── reports/                      # metricas.json, grade de hiperparâmetros, métricas CSV
```

## 1. Obter os dados do INMET
Os CSVs usados já estão em `data/raw/`. Para baixá-los novamente:
1. Acesse `https://tempo.inmet.gov.br/TabelaEstacoes/A701` (Tabela de Dados das Estações → Automáticas → SP → SAO PAULO – MIRANTE (A701)).
2. Gere a tabela em intervalos de **até 6 meses** (limite do portal): 17/05/2023–17/11/2023, 17/11/2023–17/05/2024, … até 17/11/2026.
3. Baixe cada intervalo em CSV e salve em `data/raw/`. Sobreposições de um dia entre arquivos são removidas automaticamente.

Alternativa: pacotes anuais em `https://portal.inmet.gov.br/dadoshistoricos` (ZIP de todas as estações; extraia o CSV da
A701 e coloque em `data/raw/`). O leitor aceita esse formato (latin‑1 com linhas de metadados); **ZIPs não devem ser versionados**.
Os dados são públicos e gratuitos; os horários são em UTC.

## 2. Reproduzir o treinamento
```bash
cd ml
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/nbtools.py run notebook_sereno_calor.ipynb  # ou abra no Jupyter e "Run All"
python -m pytest -q tests
```
`random_state=42` fixo: a execução reproduz o modelo, o limiar e as métricas do relatório. O notebook recria
`data/processed/`, `models/`, `figures/` e `reports/`.

## 3. Previsão e integração com a API .NET
Variáveis de ambiente:

| Variável | Uso |
|---|---|
| `SERENO_API_URL` | URL base do backend (ex.: `https://localhost:7001`); o script faz `POST {URL}/api/previsoes` |
| `SERENO_API_TOKEN` | token Bearer (opcional) |
| `SERENO_API_TIMEOUT` | timeout em segundos (padrão 10) |
| `SERENO_MODEL_PATH` | caminho alternativo do `.joblib` |

```bash
# Modo REPLAY — demonstração retrospectiva sobre o período de teste (NÃO é previsão operacional)
python predict_publish.py --replay 2025-12-24 --dry-run

# Modo OPERACIONAL — observações de hoje, ontem e anteontem
python predict_publish.py --input entrada.json
# ou a partir de CSV(s) recém-exportados da Tabela de Dados das Estações (últimos 3 dias completos)
python predict_publish.py --csv ultimos_dias_A701.csv
```

`entrada.json` (valores diários do dia civil local; faixas físicas são validadas):
```json
{
  "dataBase": "2026-01-20",
  "hoje": {"tmax": 33.5, "tmin": 20.1, "tmean": 26.0, "umi_min": 30, "umi_mean": 60,
           "chuva_mm": 0, "radiacao_mj": 24.0, "vento_mean": 1.6, "pres_mean": 924.0},
  "ontem": {"tmax": 32.8, "tmean": 25.4},
  "anteontem": {"tmax": 31.9}
}
```
(valores ilustrativos de formato, não observações reais)

### Contrato JSON enviado
```json
{
  "dataReferencia": "2025-12-25",
  "dataBase": "2025-12-24",
  "probabilidade": 0.8847,
  "nivel": "Perigo",
  "alertaExtremo": true,
  "limiarCalorC": 32.9,
  "versaoModelo": "sereno-rf-1.0",
  "estacao": "A701",
  "modo": "replay",
  "geradoEm": "2026-11-17T13:45:47+00:00",
  "observacao": "Demonstração retrospectiva com dados históricos do INMET; não é previsão meteorológica operacional."
}
```
* `dataReferencia` = dia previsto (D+1); `dataBase` = último dia observado (D).
* `nivel`: `Perigo` (p > 0,70 — “Amanhã: calor perigoso”), `Atencao` (0,40–0,70 — “Amanhã pode ser muito quente”), `SemAlerta` (< 0,40).
* `alertaExtremo`: classificação binária com o limiar de decisão 0,50 (escolhido na validação cruzada para Recall ≥ 0,80).
* `probabilidade` é a saída do classificador, **não** um índice de calor em °C.
* `observacao` só aparece no modo replay.

Sugestão de DTO no backend (ajuste ao contrato real da API):
```csharp
public record PrevisaoCalorDto(DateOnly DataReferencia, DateOnly DataBase, double Probabilidade,
    string Nivel, bool AlertaExtremo, double LimiarCalorC, string VersaoModelo,
    string Estacao, string Modo, DateTimeOffset GeradoEm, string? Observacao);
```

Erros: 4xx da API não são repetidos; falhas de rede/5xx têm 3 tentativas com backoff. Códigos de saída:
`0` ok · `2` entrada inválida · `3` falha ao publicar · `4` modelo indisponível. Logs vão para stderr.

## 4. Limitações
Uma única estação, limiar estatístico (não oficial), apenas 21 dias extremos no teste e nenhuma previsão numérica de
tempo como entrada. O modo replay não substitui dados operacionais atualizados. Veja a seção 13 do relatório.
