# Sereno SP — Dados Meteorológicos (INMET Estação A701)

Este diretório contém os dados históricos reais e processados da estação meteorológica automática **A701 – SÃO PAULO – MIRANTE (Mirante de Santana)** do Instituto Nacional de Meteorologia (INMET), utilizados no treinamento e na avaliação do modelo de previsão de calor extremo do Sereno SP.

---

## 1. Fonte dos Dados e Período

* **Estação:** A701 — SÃO PAULO – MIRANTE (Latitude: -23.496388°, Longitude: -46.619999°, Altitude: 792,06 m).
* **Período Coberto:** 17/05/2023 a 17/11/2026 (7 semestres / 42 meses / 1.282 dias de calendário).
* **Data de Extração / Download:** 17/11/2026.
* **Fontes Oficiais e Links:**
  1. **Sistema TEMPO — Tabela de Dados das Estações:**  
     [https://tempo.inmet.gov.br/TabelaEstacoes/A701](https://tempo.inmet.gov.br/TabelaEstacoes/A701)  
     *(Interface interativa que limita cada consulta a no máximo 6 meses por exportação).*
  2. **Portal de Dados Históricos do INMET (Pacotes Anuais):**  
     [https://portal.inmet.gov.br/dadoshistoricos](https://portal.inmet.gov.br/dadoshistoricos)  
     *(Pacotes anuais consolidados em ZIP contendo todas as estações do Brasil).*

> ⚠️ **Aviso Oficial do INMET sobre Dados Brutos:**  
> *"Os dados divulgados pelo INMET provenientes de estações meteorológicas automáticas são dados brutos, coletados automaticamente e transmitidos em tempo quase real, sem controle de qualidade manual prévio. Podem conter falhas instrumentais, valores sentinela (-9999), ruídos ou ausências temporárias devido a manutenção ou problemas de transmissão."*

---

## 2. Estrutura do Diretório

```text
data/
├── README.md                  # Este documento de proveniência e dicionário de dados
├── raw/                       # 7 CSVs semestrais brutos oficiais do INMET A701
│   ├── estação_sao_paulo-mirante_a701-17.05.2023-17.11.2023.csv
│   ├── estação_sao_paulo-mirante_a701-17.11.2023-17.05.2024.csv
│   ├── estação_sao_paulo-mirante_a701-17.05.2024-17.11.2024.csv
│   ├── estação_sao_paulo-mirante_a701-17.11.2024-17.05.2025.csv
│   ├── estação_sao_paulo-mirante_a701-17.05.2025-17.11.2025.csv
│   ├── estação_sao_paulo-mirante_a701-17.11.2025-17.05.2026.csv
│   └── estação_sao_paulo-mirante_a701-17.05.2026-17.11.2026.csv
└── processed/
    └── diario_a701.csv        # Base diária agregada e saneada (gerada pelo pipeline)
```

---

## 3. Arquivos Brutos Semestrais

Os 7 arquivos semestrais exportados diretamente do sistema TEMPO do INMET cobrem o histórico contínuo:

| Arquivo Exportado | Intervalo Coberto (UTC) |
|---|---|
| `estação_sao_paulo-mirante_a701-17.05.2023-17.11.2023.csv` | 17/05/2023 a 17/11/2023 |
| `estação_sao_paulo-mirante_a701-17.11.2023-17.05.2024.csv` | 17/11/2023 a 17/05/2024 |
| `estação_sao_paulo-mirante_a701-17.05.2024-17.11.2024.csv` | 17/05/2024 a 17/11/2024 |
| `estação_sao_paulo-mirante_a701-17.11.2024-17.05.2025.csv` | 17/11/2024 a 17/05/2025 |
| `estação_sao_paulo-mirante_a701-17.05.2025-17.11.2025.csv` | 17/05/2025 a 17/11/2025 |
| `estação_sao_paulo-mirante_a701-17.11.2025-17.05.2026.csv` | 17/11/2025 a 17/05/2026 |
| `estação_sao_paulo-mirante_a701-17.05.2026-17.11.2026.csv` | 17/05/2026 a 17/11/2026 |

*Formato verificado:* UTF-8 com BOM (`\xef\xbb\xbf`), sem linhas de metadados, separador `;`, decimal com vírgula `,`, valores entre aspas, datas em `dd/mm/aaaa` e horas em `HHMM` (UTC).

---

## 4. Dicionário de Variáveis Originais

### 4.1. Colunas Utilizadas no Pipeline

| Coluna Original INMET | Campo Interno | Unidade | Descrição e Agregação Diária |
|---|---|---|---|
| `Data` | `data` | `dd/mm/aaaa` | Data do registro horário (UTC) |
| `Hora (UTC)` | `hora_utc` | `HHMM` | Hora cheia do registro (0000 a 2300 UTC) |
| `Temp. Ins. (C)` | `temp_ins` | °C | Temperatura instantânea horária $\rightarrow$ Média diária (`tmean`) |
| `Temp. Max. (C)` | `temp_max` | °C | Temperatura máxima horária $\rightarrow$ Máxima diária (`tmax`) |
| `Temp. Min. (C)` | `temp_min` | °C | Temperatura mínima horária $\rightarrow$ Mínima diária (`tmin`) |
| `Umi. Ins. (%)` | `umi_ins` | % | Umidade relativa instantânea $\rightarrow$ Média diária (`umi_mean`) |
| `Umi. Min. (%)` | `umi_min` | % | Umidade relativa mínima horária $\rightarrow$ Mínima diária (`umi_min`) |
| `Umi. Max. (%)` | `umi_max` | % | Umidade relativa máxima horária (controle de consistência) |
| `Pto Orvalho Ins. (C)` | `orv_ins` | °C | Ponto de orvalho instantâneo (controle de consistência) |
| `Pto Orvalho Max. (C)` | `orv_max` | °C | Ponto de orvalho máximo horário |
| `Pto Orvalho Min. (C)` | `orv_min` | °C | Ponto de orvalho mínimo horário |
| `Pressao Ins. (hPa)` | `pres_ins` | hPa | Pressão atmosférica na estação $\rightarrow$ Média diária (`pres_mean`) |
| `Pressao Max. (hPa)` | `pres_max` | hPa | Pressão máxima horária (controle de consistência) |
| `Pressao Min. (hPa)` | `pres_min` | hPa | Pressão mínima horária (controle de consistência) |
| `Vel. Vento (m/s)` | `vento_vel` | m/s | Velocidade do vento $\rightarrow$ Média diária (`vento_mean`) |
| `Radiacao (KJ/m²)` | `radiacao_kj` | kJ/m² | Radiação solar global horária $\rightarrow$ Acumulada diária em MJ/m² (`radiacao_mj` = soma ÷ 1.000) |
| `Chuva (mm)` | `chuva_mm` | mm | Precipitação horária acumulada $\rightarrow$ Soma diária (`chuva_mm`) |

### 4.2. Colunas Descartadas e Justificativa

| Coluna Original | Motivo do Descarte |
|---|---|
| `Dir. Vento (m/s)` | Rótulo inadequado do INMET: a coluna registra a direção do vento em graus azimutais (0°–360°), não velocidade. É uma grandeza angular circular ruidosa na escala diária média, sem ganho preditivo direto para extremos de temperatura. |
| `Raj. Vento (m/s)` | Rajada máxima instantânea é fortemente correlacionada com a velocidade média do vento horária e apresenta maior sensibilidade a ruídos locais de curto prazo. |
| Linhas de metadados de cabeçalho | Metadados institucionais (código da estação, altitude, latitude) são constantes e descartados na leitura. |

---

## 5. Regras de Limpeza, Validação e Agregação

1. **Conversão de Fuso Horário:**  
   Os timestamps UTC são convertidos para o fuso local `America/Sao_Paulo` (UTC-3). O dia meteorológico analisado corresponde rigorosamente ao **dia civil local** (00h00 às 23h59 horário de Brasília).
2. **Remoção de Duplicatas:**  
   As consultas semestrais no portal do INMET geram 1 dia de sobreposição entre arquivos adjacentes (144 registros horários duplicados). Todas as duplicatas são verificadas quanto a conflitos e desduplicadas por `ts_utc`.
3. **Tratamento de Sentinelas e Faixas Físicas:**  
   Valores `-9999` e valores fora de faixas físicas aceitas (temperatura entre -10 °C e 50 °C; umidade entre 0% e 100%; pressão entre 850 hPa e 1050 hPa; radiação entre 0 e 5000 kJ/m²) são convertidos para `NaN`. Inconsistências físicas (`temp_min > temp_max`) são anuladas.
4. **Critério de Exclusão de Dias Incompletos (> 25% de ausência):**  
   Dias com **menos de 18 horas válidas de temperatura** (menos de 75% de cobertura) são marcados como `dia_valido = False` e suas variáveis ficam como `NaN`. **Nenhum valor é imputado artificialmente**.
5. **Controle de Qualidade em Radiação e Chuva:**  
   A ausência noturna de radiação é esperada fisicamente (valor 0 ou vazio). Lacunas no período diurno (10h às 15h) com temperatura válida invalidam o acumulado diário de radiação daquele dia. Chuva diária é invalidada se houver menos de 18 horas de registros de precipitação.

---

## 6. Passo a Passo para Baixar e Reproduzir

1. Acesse o portal do INMET: [https://tempo.inmet.gov.br/TabelaEstacoes/A701](https://tempo.inmet.gov.br/TabelaEstacoes/A701).
2. No menu, selecione **Estação: SAO PAULO - MIRANTE (A701)**.
3. Defina os intervalos de datas semestrais (máximo 6 meses por consulta):
   - `17/05/2023` a `17/11/2023`
   - `17/11/2023` a `17/05/2024`
   - `17/05/2024` a `17/11/2024`
   - `17/11/2024` a `17/05/2025`
   - `17/05/2025` a `17/11/2025`
   - `17/11/2025` a `17/05/2026`
   - `17/05/2026` a `17/11/2026`
4. Clique em **Gerar Tabela** e em **Baixar CSV**.
5. Salve os arquivos na pasta `data/raw/`.
6. Execute o pipeline no notebook ou via script para reconstruir a base `data/processed/diario_a701.csv`.
