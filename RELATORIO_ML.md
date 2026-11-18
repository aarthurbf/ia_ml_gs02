# Sereno SP — Relatório Técnico de Machine Learning
**Previsão de Calor Extremo na Véspera (D+1) · Estação INMET A701 (Mirante de Santana)**  
**Global Solution 2026 · FIAP · Inteligência Artificial & Machine Learning**

| Integrantes | RM |
|---|---|
| Arthur Bobadilla Franchi | RM555056 |
| Luan Orlandelli Ramos | RM554747 |
| Jorge Luis | RM554418 |

* **Repositório GitHub:** [https://github.com/aarthurbf/ia_ml_gs02](https://github.com/aarthurbf/ia_ml_gs02)
* **Vídeo Demonstrativo (YouTube Não Listado):** [https://youtu.be/WlFBxHZtbzQ](https://youtu.be/WlFBxHZtbzQ)

---

## 1. Problema Social e Formulação de Machine Learning
O distrito de Santana (Zona Norte de São Paulo) abriga expressiva população idosa (60+) e trabalhadores ao ar livre sujeitos a estresse térmico severo sem aviso prévio. O **Sereno SP** antecipa dias de calor extremo **com 24 horas de antecedência**, permitindo hidratação orientada, realocação de turnos de trabalho e monitoramento comunitário.

* **Pergunta de ML:** *Com base nas condições meteorológicas observadas hoje ($D$) e nos dois dias anteriores ($D-1, D-2$) na estação A701, amanhã ($D+1$) será um dia de calor extremo?*
* **Formulação:** Classificação binária supervisionada ($y \in \{0, 1\}$) com probabilidade calibrada $P(y=1|X) \in [0, 1]$.

---

## 2. Dados Reais, Pré-processamento e Divisão Temporal

### 2.1. Proveniência e Coleta
* **Fonte Oficial:** INMET — Tabela de Dados das Estações ([https://tempo.inmet.gov.br/TabelaEstacoes/A701](https://tempo.inmet.gov.br/TabelaEstacoes/A701)), dados públicos.
* **Estação & Período:** A701 (São Paulo – Mirante), 17/05/2023 a 17/11/2026 (7 semestres / 1.282 dias).
* **Estrutura Bruta:** 30.888 registros horários em UTF-8 BOM, sem metadados, separador `;` e decimal `,`.

### 2.2. Engenharia de Dados e Limpeza Rigorosa (Sem Imputação)
1. **Fuso Horário:** Conversão UTC $\rightarrow$ `America/Sao_Paulo` (UTC-3), agregando pelo dia civil local.
2. **Duplicatas e Sentinelas:** 144 duplicatas de sobreposição semestral removidas. Zero valores $-9999$.
3. **Critério de Exclusão:** Dias com $> 25\%$ de horas ausentes ($< 18$ h válidas de temperatura) foram excluídos da modelagem (**18 dias descartados**). Radiação anulada em 10 dias com falha diurna (10h–15h); chuva anulada em 20 dias. **Nenhum valor foi preenchido artificialmente.**
4. **Atributos (15 features):** $T_{max}, T_{min}, T_{mean}$, umidade mín./média, radiação, chuva, vento, pressão e amplitude térmica de $D$; defasagens $T_{max}, T_{mean}$ de $D-1$ e $T_{max}$ de $D-2$; sazonalidade cíclica ($\sin, \cos$ do mês). Deslocamentos por data contínua de calendário (sem vazamento).

### 2.3. Variável-Alvo e Divisão Temporal Estrita
* **Alvo:** $extremo\_amanha = 1$ se $T_{max}(D+1) \ge P_{90}$ das máximas diárias do **treino** = **32,9 °C** (902 dias válidos de treino). Critério estatístico local sem contaminação do teste.
* **Divisão Cronológica:**
  - **Treino:** 19/05/2023 a 15/11/2025 (alvo até 16/11/2025; $N=846$, 88 positivos / 10,40%; 68 linhas descartadas por defasagens).
  - **Teste (Hold-out):** 17/11/2025 a 15/11/2026 (alvo até 16/11/2026; $N=349$, 21 positivos / 6,02%; 15 descartadas).
  - *Justificativa:* O teste isola exatamente 12 meses contínuos (um ciclo sazonal completo com o verão 25/26), sem qualquer sobreposição temporal com o treino.

---

## 3. Escolha do Modelo e Otimização de Hiperparâmetros

### 3.1. Justificativa do Algoritmo
Optou-se por **Random Forest Classifier** (`class_weight='balanced'`) pela capacidade de capturar interações não lineares (ex.: radiação solar $\times$ estação do ano), robustez a variáveis em escalas distintas sem necessidade de normalização e interpretabilidade direta via importância de variáveis.

### 3.2. Validação Cruzada Temporal (TimeSeriesSplit com 5 Dobras no Treino)
O ajuste foi conduzido via `GridSearchCV` avaliando 12 combinações com métrica de seleção $F_1$-Score da classe minoritária.

| $n\_estimators$ | $max\_depth$ | $min\_samples\_leaf$ | $F_1$ Validação (Média $\pm$ DP) | $F_1$ Treino (Dobras) | Rank |
|---:|---|---:|---|---:|:---:|
| **300** | **None** | **5** | **0,553 $\pm$ 0,060** | **0,777** | **1** |
| 300 | 10 | 5 | 0,550 $\pm$ 0,068 | 0,775 | 2 |
| 100 | None | 5 | 0,538 $\pm$ 0,084 | 0,765 | 3 |
| 100 | 10 | 5 | 0,538 $\pm$ 0,084 | 0,768 | 4 |
| 300 | 5 | 5 | 0,535 $\pm$ 0,080 | 0,733 | 5 |
| 300 | 5 | 1 | 0,505 $\pm$ 0,074 | 0,799 | 6 |
| 100 | 5 | 5 | 0,503 $\pm$ 0,092 | 0,729 | 7 |
| 100 | 5 | 1 | 0,463 $\pm$ 0,162 | 0,802 | 8 |
| 100 | 10 | 1 | 0,459 $\pm$ 0,099 | 0,961 | 9 |
| 100 | None | 1 | 0,384 $\pm$ 0,218 | 1,000 | 10 |
| 300 | 10 | 1 | 0,376 $\pm$ 0,197 | 0,967 | 11 |
| 300 | None | 1 | 0,369 $\pm$ 0,217 | 1,000 | 12 |

* **Configuração Vencedora:** $n\_estimators=300, max\_depth=None, min\_samples\_leaf=5$.
* **Limiar de Decisão Operacional:** Fixado em **0,51** a partir da curva Precision-Recall das probabilidades fora da amostra (OOF) do treino, assegurando a meta de **Recall $\ge 0,80$** (Recall OOF = 0,809; Precision OOF = 0,466).

---

## 4. Resultados Treino × Teste e Comparação com Baselines

| Modelo / Conjunto | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | VP | FP | FN | VN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF — Treino (In-sample) | 0,941 | 0,648 | 0,943 | 0,769 | 0,989 | 0,918 | 83 | 45 | 5 | 713 |
| **RF — Teste (Hold-out)** | **0,957** | **0,607** | **0,810** | **0,694** | **0,966** | **0,782** | **17** | **11** | **4** | **317** |
| Persistência — Teste ($T_{max}(D) \ge 32,9$) | 0,966 | 0,714 | 0,714 | 0,714 | 0,848 | 0,527 | 15 | 6 | 6 | 322 |
| Climatológico — Teste (Moda = 0) | 0,940 | 0,000 | 0,000 | 0,000 | 0,500 | 0,060 | 0 | 0 | 21 | 328 |

### 4.1. Análise Crítica dos Baselines
* **Persistência:** Apresenta $F_1$ ligeiramente superior (0,714 vs. 0,694) devido à forte inércia térmica em São Paulo. No entanto, o **Random Forest antecipa 2 dias extremos a mais** (Recall 0,810 vs. 0,714; 17 vs. 15 acertos) e possui capacidade discriminativa probabilística muito superior (**PR-AUC de 0,782 vs. 0,527** e ROC-AUC de 0,966 vs. 0,848), viabilizando níveis escalonados de alerta.
* **Climatológico:** Atinge 94% de acurácia sem alertar nenhum dia crítico, evidenciando por que acurácia é inadequada para eventos raros.

---

## 5. Diagnóstico de Overfitting e Curva de Validação
* **Diferença Treino × Teste:** $\Delta F_1 = F_{1,treino} - F_{1,teste} = 0,769 - 0,694 = \mathbf{0,075}$, dentro da tolerância rigorosa de $\le 0,10$.
* **Curva de Validação ($max\_depth$):** Na análise isolada de profundidade (gerada em `figures/09_curva_validacao_max_depth.png`), observa-se que com $min\_samples\_leaf=1$, o modelo memoriza o treino ($F_1 \rightarrow 1,00$) e degrada na validação ($F_1 \rightarrow 0,37$). A exigência de no mínimo 5 amostras por folha ($min\_samples\_leaf=5$) regulariza a estrutura das árvores, estabilizando o $F_1$ de validação em $\approx 0,553$ mesmo sem limitar a profundidade máxima.

---

## 6. Interpretação das Variáveis e Leitura para o Cidadão

### 6.1. Relevância Meteorológica
As variáveis de maior impacto na previsão foram:
1. $T_{max}$ de hoje (Gini: 0,232; Permutação: 0,244; SHAP: 0,120)
2. Radiação Solar acumulada hoje (Gini: 0,151; Permutação: 0,095; SHAP: 0,072)
3. $T_{mean}$ de hoje (Gini: 0,105; Permutação: 0,091)
4. Sazonalidade ($\cos(m\hat{e}s)$) e Umidade Mínima de hoje.

*Interpretação:* Dias de céu limpo, alta insolação e ar seco na primavera/verão mantêm a massa de ar aquecida sobre a bacia de São Paulo, predispondo o dia seguinte ao calor extremo.

### 6.2. Tradução Prática para os Moradores e Trabalhadores de Santana
* **Sensibilidade (Recall 0,81):** De cada 10 dias de calor extremo no teste, o sistema **alertou 8 na véspera** (17 de 21 dias).
* **Confiabilidade (Precision 0,61):** De cada 10 alertas emitidos, **6 foram confirmados** como calor extremo ($T_{max} \ge 32,9$ °C). Nos 11 casos de falso alarme, a temperatura observada foi quente ($29,4$ a $31,8$ °C), não gerando desmobilização inútil.
* **Comunicação por Níveis de Risco no Teste:**
  - **`Perigo` ($P > 0,70$):** 20 alertas emitidos $\rightarrow$ **15 confirmados (75,0% de precisão)**.
  - **`Atenção` ($0,40 \le P \le 0,70$):** 16 alertas $\rightarrow$ 3 confirmados (18,8%; probabilidade 3x superior à média basal).
  - **`Sem Alerta` ($P < 0,40$):** 313 dias $\rightarrow$ apenas 3 eventos não antecipados (1,0% de erro omissivo).

---

## 7. Limitações e Trabalhos Futuros
1. **Representatividade Espacial:** Uso de estação única (A701 Mirante de Santana, 792 m). Ilhas de calor em fundos de vale e áreas densamente pavimentadas podem vivenciar temperaturas superiores.
2. **Critério Estatístico:** Limiar de 32,9 °C baseado no $P_{90}$ local, não incluindo umidade na definição da variável-alvo (índice de calor/sensação térmica).
3. **Amostragem no Teste:** Período de teste com 21 dias positivos (baixa ocorrência em 2026).
4. **Próximos Passos:** Incorporação de saídas de modelos numéricos de previsão de tempo (NWP/GFS) como covariáveis e expansão para a rede de telemetria do CGE-SP.
