# Roteiro do vídeo — Sereno SP · IA & ML (até 5 min)
Todos os números vêm da execução registrada em `notebook_sereno_calor.ipynb` / `reports/metricas.json`.

## 0:00–0:40 · Problema e pergunta de ML (40 s)
**Tela:** foto/mapa de Santana + título do projeto.
**Fala:** “Idosos e trabalhadores de rua de Santana enfrentam o calor extremo sem aviso. O Sereno SP quer avisar na
véspera. Nossa pergunta de Machine Learning é: com o tempo observado hoje e nos dois dias anteriores na estação do INMET
no Mirante de Santana, amanhã será um dia de calor extremo? O modelo responde sim ou não e dá uma probabilidade.”

## 0:40–1:40 · Dados e preparação (60 s)
**Tela:** `01_serie_tmax_limiar.png`, tabela de limpeza do notebook.
**Fala:** “Usamos dados reais e públicos do INMET, estação A701: mais de 30 mil registros horários, de maio de 2023 a
novembro de 2026. Removemos 144 horas duplicadas, convertemos UTC para o horário de São Paulo e excluímos 18 dias com
mais de 25% das horas ausentes — sem inventar nenhum valor. Transformamos tudo em dados diários e criamos 15 variáveis
só com o que se sabe até hoje. Dia de calor extremo é aquele com máxima igual ou acima do percentil 90 do treino:
32,9 graus, calculado só com dados de treino para evitar vazamento.”

## 1:40–2:40 · Modelo e otimização (60 s)
**Tela:** `03_heatmap_hiperparametros.png`.
**Fala:** “Escolhemos Random Forest: funciona bem com dados em tabela, capta relações não lineares e mostra quais variáveis
pesam mais. Separamos o tempo em ordem: treino de 2023 a novembro de 2025 e teste no ano seguinte. Testamos 12
combinações com validação cruzada temporal de 5 dobras, só no treino, usando o F1 dos dias extremos. A melhor:
300 árvores, sem limite de profundidade e no mínimo 5 dias por folha — F1 médio de 0,555. Folhas de 1 dia decoravam o
treino e pioravam a validação.”

## 2:40–4:00 · Resultados de treino e teste (80 s)
**Tela:** `07_treino_vs_teste.png`, `08_rf_vs_baselines.png`, `04_matriz_confusao_teste.png`.
**Fala:** “No treino, F1 de 0,76; no teste, 0,69 — diferença de 0,065, abaixo do nosso alerta de overfitting de 0,10. No
teste houve 21 dias extremos: o modelo acertou 17 e deixou passar 4, com 11 alarmes falsos. Comparamos com dois modelos
simples. Sempre dizer ‘não’ tem 94% de acerto e não avisa ninguém — por isso acurácia engana. A persistência, ‘amanhã
será como hoje’, é um adversário forte: F1 de 0,71, um pouco acima do nosso 0,69. O Random Forest antecipa 2 dias
extremos a mais, com recall de 0,81 contra 0,71, e ordena muito melhor o risco: PR‑AUC de 0,78 contra 0,53. Com só
21 dias extremos, essas diferenças ainda são pequenas.”

## 4:00–5:00 · Leitura para o cidadão e limitações (60 s)
**Tela:** `06_importancia_variaveis.png` + JSON do modo replay de 24/12/2025 (nível “Perigo”, 0,88; máxima observada no dia seguinte 35,9 °C).
**Fala:** “Em linguagem simples: de cada 10 dias de calor extremo, o Sereno SP antecipou 8. De cada 10 alertas, 6 se
confirmaram — e nos outros a máxima ainda passou de 29 graus. Quando o aviso foi ‘calor perigoso’, 15 de 20 se
confirmaram. O que mais pesa é a máxima de hoje, a radiação solar e a época do ano — isso é associação, não causa.
Limitações: uma só estação, limiar estatístico e poucos dias extremos no teste. A demonstração que mostramos é um
replay histórico; para operar de verdade, o sistema precisa receber os dados do dia. O próximo passo é incluir
previsões numéricas do tempo e mais anos de histórico.”
