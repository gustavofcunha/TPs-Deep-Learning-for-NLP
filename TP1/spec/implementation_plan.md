# Plano de Trabalho: Implementação Gradual do TP1

Este plano detalha o passo a passo para a implementação das seções do notebook (`TP1/src/word2vec_hyperparameters.ipynb`), com base nas decisões aprovadas e nas diretrizes pedagógicas.

---

## 1. O que Estamos Calculando Exatamente? (Métricas Estatísticas e Escala Mapeada)

Para cada pergunta do teste de analogias $i$ (ex.: *Paris : France :: Berlin : Germany*):
1. **Cálculo do Vetor Resultante $\vec{R}_i$ (Decisão 1 - Fórmula Literal do Enunciado):**
   $$\vec{R}_i = \vec{v}(\text{France}) + \vec{v}(\text{Paris}) - \vec{v}(\text{Berlin})$$
2. **Distância de Cosseno Original ($d_i \in [0, 2]$):**
   $$d_i = 1 - \frac{\vec{R}_i \cdot \vec{v}(\text{Germany})}{\|\vec{R}_i\|_2 \|\vec{v}(\text{Germany})\|_2} \quad \in [0, 2]$$
3. **Mapeamento de Alinhamento Intuitivo ($s_i \in [-1, 1]$):**
   Para tornar a interpretação e a visualização intuitivas, mapeamos linearmente a distância para a escala de alinhamento:
   $$s_i = 1 - d_i = \frac{\vec{R}_i \cdot \vec{v}(\text{Germany})}{\|\vec{R}_i\|_2 \|\vec{v}(\text{Germany})\|_2} \quad \in [-1, 1]$$
   - $s_i = +1.0$: Alinhamento perfeito na mesma direção ($0^\circ$).
   - $s_i = 0.0$: Vetores ortogonais / neutros ($90^\circ$).
   - $s_i = -1.0$: Vetores em direções opostas ($180^\circ$).
   - **Interpretação:** Quanto maior, melhor. Todos os gráficos comparativos, mapas de calor e análises usarão essa escala $[-1, 1]$.

A partir desses valores individuais, calculamos as seguintes **estatísticas consolidadas para cada um dos 54 modelos**:

| Métrica | O que estamos calculando? | Interpretação Científica |
| :--- | :--- | :--- |
| **Média Global de Alinhamento ($\overline{S}_{\text{global}}$)** | A média simples de todos os $s_i$ nas perguntas cobertas. | **Métrica principal:** Varia de -1 a 1. Quanto maior, melhor a capacidade preditiva do modelo. |
| **Desvio Padrão Global ($\sigma_{\text{global}}$)** | A variabilidade dos acertos individuais em torno da média. | **Estabilidade:** Um desvio menor significa comportamento homogêneo e previsível entre as perguntas. |
| **Médias por Categoria ($\overline{S}_1, \dots, \overline{S}_{14}$)** | A média de alinhamento calculada separadamente para cada uma das 14 categorias. | Diagnóstico fino: revela os pontos fortes e fracos específicos do modelo. |
| **Desvio Padrão entre Categorias ($\sigma_{\text{categorias}}$)** | O desvio padrão das 14 médias de categorias. | **Consistência temática:** Se for baixo, o modelo é versátil; se for alto, ele é especialista em um domínio e falho em outro. |
| **Média Semântica vs. Sintática** | A média das categorias de significado comparada à média das categorias gramaticais. | Compara se a configuração favorece relações semânticas ou sintáticas. |
| **Distância de Cosseno ($\overline{D}_{\text{global}}, \sigma_{\text{global}}$)** | A métrica original em $[0, 2]$ onde menor é melhor. | Preservada para conformidade estrita com o texto do enunciado do professor. |

---

## 2. Consumo de Recursos Computacionais e Informações de Hardware

Para avaliar o *trade-off* entre ganho de desempenho e custo computacional (fronteira de Pareto), registramos a cada modelo:
- **Perfil do Hardware:** Modelo da CPU, quantidade de núcleos lógicos, memória RAM total disponível e acelerador GPU (quando presente no Colab).
- **Paralelismo:** Número de *worker threads* ativas (`NUM_WORKERS = max(1, cpu_count)`).
- **Tempo de Execução:** Duração do treinamento (`train_time_sec`), tempo de avaliação (`eval_time_sec`) e tempo total do passo.
- **Tempo de CPU:** Tempo efetivo de processador acumulado (`cpu_time_sec`).
- **Consumo de Memória:** Pico de uso de memória física residente (`peak_ram_mb`).
- **Vazão (*Throughput*):** Milhares de palavras processadas por segundo (`throughput_kwords_sec`).

---

## 3. Como Evitar Perder Tudo se o Colab Desconectar? (Mecanismo de Resiliência)

1. **Checkpoint Imediato em Disco (`TP1/outputs/experiment_results.csv`):**
   * Assim que o modelo $k$ termina o treino e a avaliação, a linha completa com seus hiperparâmetros, métricas e dados de hardware é gravada no CSV.
   * Não mantemos resultados exclusivamente na memória volátil (RAM).
2. **Capacidade de Retomada Automática (*Resume*):**
   * Antes de treinar o modelo $k$, o código verifica se a combinação já está presente no CSV.
   * Se já estiver salva, pula instantaneamente para a próxima.
   * Se a sessão cair no meio da grade, basta reconectar e rodar a célula novamente: ela continua exatamente de onde parou.
3. **Barra de Progresso e Logs de Execução:**
   * Prints formatados indicando o progresso `[k/54]`, o tempo do passo, o pico de RAM e o horário projetado de término (ETA).

---

## 4. Roteiro Gradual de Implementação

### Etapa 1: Função Modular de Treinamento (Seção 3) - CONCLUÍDA
* Implementação da função `train_word2vec_model(...)` com reprodutibilidade (`seed=42`) e paralelismo em CPU.
* Validação via *smoke test* rápido no Colab (executado em 53s).

### Etapa 2: Módulo de Avaliação por Álgebra Vetorial e Estatística (Seção 4) - CONCLUÍDA NO NOTEBOOK
* Implementação da função `evaluate_word2vec_analogies(model, df_analogies)` com suporte duplo: Distância de Cosseno $[0, 2]$ e Score de Alinhamento $[-1, 1]$.
* Cabeçalho justificando em primeira pessoa o uso da escala intuitiva $[-1, 1]$ para todos os gráficos e análises.
* Célula de teste validada no *smoke model* com 91,21% de cobertura das analogias.

### Etapa 3: Execução Controlada da Grade de 54 Modelos (Seção 5) - IMPLEMENTADA NO NOTEBOOK
* Vetores de busca parametrizáveis (`GRID_ARCHITECTURES`, `GRID_WINDOWS`, `GRID_VECTOR_SIZES`, `GRID_EPOCHS`).
* Profiling de hardware e consumo de recursos (CPU, RAM, throughput, tempo).
* Logs dinâmicos de execução (progresso, tempo decorrido, ETA dinâmico e previsão de término), sem poluição de scores no terminal.
* Gravação incremental em `TP1/outputs/experiment_results.csv`.

### Etapa 4: Visualizações e Conclusão Científica (Seções 6 e 7)
* Gráficos comparativos com a escala de alinhamento $[-1, 1]$:
  1. Comparação CBOW vs. Skip-gram (barras com intervalo de desvio padrão).
  2. Efeito do Tamanho da Janela (curvas de alinhamento por $w \in \{2, 5, 10\}$).
  3. Heatmap de Dimensão do Vetor $\times$ Épocas.
  4. Curva de *Trade-off* de Pareto: Desempenho (Alinhamento) $\times$ Tempo de Treinamento e Memória.
  5. Projeção geométrica 2D (PCA / t-SNE) das analogias.
  6. Discussão e síntese identificando a melhor combinação.
