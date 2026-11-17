# Sereno SP — Módulo de IA & Machine Learning
Previsão, na véspera, de **dia de calor extremo** em São Paulo (estação INMET **A701 – Mirante de Santana**) com **Random Forest** (scikit-learn).  
**Global Solution 2026 · FIAP · Inteligência Artificial & Machine Learning**

| Integrantes | RM |
|---|---|
| Arthur Bobadilla Franchi | RM555056 |
| Luan Orlandelli Ramos | RM554747 |
| Jorge Luis | RM554418 |

---

## Resumo dos Resultados no Período de Teste (17/11/2025 → 16/11/2026)

| Métrica / Parâmetro | Valor Obtido |
|---|---|
| **Alvo:** Limiar de Calor Extremo ($P_{90}$ do treino) | **32,9 °C** ($T_{max}$ de D+1 $\ge 32,9$ °C) |
| **Limiar de Decisão do Classificador:** | **0,51** (otimizado via PR-Curve OOF no treino para Recall $\ge 0,80$) |
| **Recall (Sensibilidade — Classe 1):** | **0,810** (17 de 21 dias extremos antecipados) |
| **Precision (Precisão — Classe 1):** | **0,607** (17 de 28 alertas emitidos confirmados) |
| **F1-Score (Classe 1):** | **0,694** |
| **ROC-AUC / PR-AUC:** | **0,966 / 0,782** |
| **Baseline de Persistência ($T_{max}$ hoje $\ge 32,9$ °C):** | F1 = **0,714** (Precision 0,714 / Recall 0,714) |
| **Baseline Climatológico (moda do treino = 0):** | F1 = **0,000** (Accuracy 0,940 sem detectar extremos) |
| **Diagnóstico de Overfitting ($F1_{treino} - F1_{teste}$):** | **0,075** ($0,769 - 0,694$; dentro do limite $\le 0,10$) |

* **Repositório GitHub:** [https://github.com/aarthurbf/ia_ml_gs02](https://github.com/aarthurbf/ia_ml_gs02)
* **Vídeo Demonstrativo:** `[LINK DO VÍDEO YOUTUBE - PENDENTE DE GRAVAÇÃO PELA EQUIPE]`

> 📖 **Relatório Técnico Completo:** [`RELATORIO_ML.md`](RELATORIO_ML.md) (≤ 3 páginas, tabelas da rubrica, análise crítica e baselines).
> 🎥 **Roteiro do Vídeo de Apresentação:** [`ROTEIRO_VIDEO.md`](ROTEIRO_VIDEO.md) (5 minutos cronometrados).
> 🤖 **Registro de Prompts de IA:** [`prompts.md`](prompts.md).

---

## Estrutura do Projeto

```text
ml/
├── notebook_sereno_calor.ipynb     # Jupyter Notebook completo, executado com todas as saídas
├── RELATORIO_ML.md                 # Relatório técnico acadêmico (máximo 3 páginas)
├── ROTEIRO_VIDEO.md                # Roteiro cronometrado de gravação do vídeo (5 min)
├── README.md                       # Guia do módulo e contrato de integração
├── prompts.md                      # Registro de uso e governança de IA
├── requirements.txt                # Dependências fixadas (Python 3.12 + jupyter)
├── predict_publish.py              # Script de inferência e publicação (POST /api/previsoes)
├── sereno_ml/                      # Pacote Python com a lógica modular
│   ├── __init__.py
│   ├── data.py                     #   Leitura, limpeza, faixas físicas e agregação diária
│   ├── features.py                 #   15 atributos temporais (sem vazamento) e variável-alvo
│   └── modeling.py                 #   Divisão temporal, baselines, métricas e limiares
├── tests/
│   └── test_pipeline.py            # 23 testes unitários e de integração (pytest)
├── tools/
│   ├── notebook_src.py             # Código-fonte reprodutível do notebook
│   └── nbtools.py                  # Construtor e executor automatizado do .ipynb
├── data/
│   ├── README.md                   # Documentação detalhada dos dados e fontes do INMET
│   ├── raw/                        # 7 CSVs semestrais brutos do INMET A701 (versionados)
│   └── processed/
│       └── diario_a701.csv         # Base diária agregada e validada (1.282 dias)
├── models/
│   └── sereno_rf.joblib            # Pacote serializado (Random Forest + metadados de inferência)
├── figures/                        # 11 gráficos em alta resolução (150 dpi)
└── reports/                        # metricas.json, grade de hiperparâmetros e métricas em CSV
```

---

## 1. Dados e Reprodutibilidade

Os dados brutos consistem em 7 arquivos semestrais da estação automática **INMET A701 – São Paulo – Mirante** (17/05/2023 a 17/11/2026), totalizando 30.888 registros horários.

* **Fonte:** Sistema TEMPO INMET ([https://tempo.inmet.gov.br/TabelaEstacoes/A701](https://tempo.inmet.gov.br/TabelaEstacoes/A701)).
* **Limpeza:** 144 duplicatas removidas, conversão para fuso local `America/Sao_Paulo`, 18 dias excluídos por ausência de dados (> 25% de horas faltantes), **nenhuma imputação artificial**.
* **Dicionário e Procedimentos:** consulte [`data/README.md`](data/README.md).

---

## 2. Como Reproduzir o Treinamento e Testes

```bash
# 1. Criar e ativar o ambiente virtual (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate       # No Windows: .venv\Scripts\activate

# 2. Instalar dependências exatas
pip install -r requirements.txt

# 3. Executar o pipeline completo via script construtor (ou abrir o notebook)
python tools/nbtools.py build tools/notebook_src.py notebook_sereno_calor.ipynb
python tools/nbtools.py run notebook_sereno_calor.ipynb

# 4. Rodar a suíte de testes automatizados
python -m pytest -v tests/test_pipeline.py
```

---

## 3. Inferência e Integração com a API .NET

O script `predict_publish.py` suporta dois modos operacionais:

1. **Modo Replay (Demonstração Retrospectiva):**  
   Simula o envio da previsão a partir dos dados já observados na base histórica de teste.
   ```bash
   python predict_publish.py --replay 2025-12-24 --dry-run
   ```
2. **Modo Operacional (Produção Diária):**  
   Recebe as observações dos últimos 3 dias via arquivo JSON ou novos CSVs do INMET:
   ```bash
   python predict_publish.py --input entrada.json
   # ou com CSV recente da estação A701
   python predict_publish.py --csv ultimos_dias_A701.csv
   ```

### Variáveis de Ambiente
* `SERENO_API_URL`: URL base da API .NET (ex.: `https://localhost:7001` ou `https://api.serenosp.fiap.com.br`).
* `SERENO_API_TOKEN`: Token de autenticação Bearer (opcional).
* `SERENO_API_TIMEOUT`: Timeout em segundos para a requisição HTTP (padrão: `10`).
* `SERENO_MODEL_PATH`: Caminho customizado para o modelo `.joblib` (padrão: `models/sereno_rf.joblib`).

---

## 4. Contrato de Integração com o Backend (.NET / C#)

### 4.1. Payload JSON Enviado (`POST /api/previsoes`)

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
  "geradoEm": "2026-11-17T18:15:00+00:00",
  "observacao": "Demonstração retrospectiva com dados históricos do INMET; não é previsão meteorológica operacional."
}
```

### 4.2. Mapeamento com as Entidades do Backend .NET

* **Entidade de Domínio:** `PrevisaoCalor.cs`
* **Enum de Níveis:** `NivelAlerta.cs` (`SemAlerta = 0`, `Atencao = 1`, `Perigo = 2`)
* **Controller:** `PrevisoesController.cs` (endpoint `POST /api/previsoes`)

```csharp
// Exemplo de DTO / Entidade no backend C#
public record PrevisaoCalorDto(
    DateOnly DataReferencia,      // Obrigatório: dia previsto (D+1)
    DateOnly DataBase,            // Dia da última observação (D)
    double Probabilidade,         // Obrigatório: P(calor extremo) entre 0.0 e 1.0
    string Nivel,                 // Obrigatório: "Perigo" | "Atencao" | "SemAlerta"
    bool AlertaExtremo,           // Classificação binária baseada no limiar de decisão (0.51)
    double LimiarCalorC,          // Limiar de temperatura em °C (32.9)
    string VersaoModelo,          // Obrigatório: versão do modelo ("sereno-rf-1.0")
    string Estacao,               // Código da estação ("A701")
    string Modo,                  // "operacional" ou "replay"
    DateTimeOffset GeradoEm,      // Timestamp UTC de geração
    string? Observacao            // Informações adicionais de auditoria / replay
);
```

### 4.3. Coerência dos Limiares de Comunicação vs. Limiar de Decisão

* **`alertaExtremo` (Classificação Binária = 0,51):**  
  Ancorado na curva Precision-Recall das probabilidades fora da amostra (OOF) do treino para garantir **Recall $\ge 0,80$** (prioridade em evitar falsos negativos para idosos e trabalhadores).
* **`nivel` (Comunicação Graduada para o Cidadão):**
  - **`SemAlerta` ($p < 0,40$):** Risco basal baixo. Menos de 1% de falsos negativos no teste.
  - **`Atencao` ($0,40 \le p \le 0,70$):** Faixa de incerteza moderada. Probabilidade 4x a 7x superior à prevalência basal. Mensagem preventiva: *"Amanhã pode ser muito quente — hidrate-se e evite exposição solar ao meio-dia"*.
  - **`Perigo` ($p > 0,70$):** Alta probabilidade e convicção. No conjunto de teste, **75,0% dos alertas emitidos neste nível se confirmaram como calor extremo real**.

### 4.4. Exemplos de Resposta da API .NET

#### Sucesso — HTTP 201 Created
```http
HTTP/1.1 201 Created
Content-Type: application/json; charset=utf-8
Location: /api/previsoes/128

{
  "id": 128,
  "dataReferencia": "2025-12-25",
  "nivel": "Perigo",
  "probabilidade": 0.8847,
  "mensagem": "Previsão registrada com sucesso e alertas disparados para a região de Santana."
}
```

#### Erro de Validação — HTTP 400 Bad Request
```http
HTTP/1.1 400 Bad Request
Content-Type: application/problem+json; charset=utf-8

{
  "type": "https://tools.ietf.org/html/rfc7231#section-6.5.1",
  "title": "One or more validation errors occurred.",
  "status": 400,
  "errors": {
    "Probabilidade": ["A probabilidade deve estar entre 0.0 e 1.0."],
    "Nivel": ["O nível deve ser 'Perigo', 'Atencao' ou 'SemAlerta'."]
  }
}
```

---

## 5. Limitações Conhecidas

1. **Estação Única:** Modelo calibrado exclusivamente para o microclima da estação A701 (Mirante de Santana, 792 m de altitude). Ilhas de calor urbanas nas avenidas de Santana podem registrar temperaturas superiores.
2. **Critério Estatístico Local:** O limiar de 32,9 °C é o $P_{90}$ da série histórica do Mirante de Santana, não um índice oficial de sensação térmica ou temperatura aparente.
3. **Volume de Positivos no Teste:** O ciclo de teste 2025/2026 apresentou 21 dias de calor extremo (prevalência de 6,02%), o que torna diferenças de 1–2 casos sensíveis em métricas pontuais.
4. **Variáveis Atmosféricas Observadas:** O modelo depende apenas de dados observados até o dia D, sem utilizar como features as previsões numéricas de modelos sinóticos (GFS/ECMWF).
