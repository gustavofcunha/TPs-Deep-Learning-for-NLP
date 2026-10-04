# Implementation Plan: Incremental Execution for TP1

This plan details the step-by-step roadmap for implementing the notebook sections (`TP1/src/word2vec_hyperparameters.ipynb`), aligned with approved decisions and pedagogical guidelines.

---

## 1. What Are We Computing Exactly? (Statistical Metrics and Scale Mapping)

For each analogy question $i$ (e.g., *Paris : France :: Berlin : Germany*):
1. **Resultant Vector Calculation $\vec{R}_i$ (Literal Assignment Formula):**
   $$\vec{R}_i = \vec{v}(\text{France}) + \vec{v}(\text{Paris}) - \vec{v}(\text{Berlin})$$
2. **Original Cosine Distance ($d_i \in [0, 2]$):**
   $$d_i = 1 - \frac{\vec{R}_i \cdot \vec{v}(\text{Germany})}{\|\vec{R}_i\|_2 \|\vec{v}(\text{Germany})\|_2} \quad \in [0, 2]$$
3. **Intuitive Alignment Mapping ($s_i \in [-1, 1]$):**
   To make interpretation intuitive, we linearly map distance to an alignment scale:
   $$s_i = 1 - d_i = \frac{\vec{R}_i \cdot \vec{v}(\text{Germany})}{\|\vec{R}_i\|_2 \|\vec{v}(\text{Germany})\|_2} \quad \in [-1, 1]$$
   - $s_i = +1.0$: Perfect collinear alignment ($0^\circ$).
   - $s_i = 0.0$: Orthogonal vectors ($90^\circ$).
   - $s_i = -1.0$: Opposing directions ($180^\circ$).
   - **Interpretation:** Higher is better. Formal comparative analyses evaluate both the assignment distance $[0, 2]$ and alignment $[-1, 1]$.

From these individual values, we compute consolidated statistics across each of the 54 models:

| Metric | Calculation Description | Scientific Interpretation |
| :--- | :--- | :--- |
| **Global Mean Alignment ($\overline{S}_{\text{global}}$)** | Arithmetic mean of all $s_i$ across evaluated quadruplets. | Global capability: ranges from -1 to 1; higher indicates superior predictive geometry. |
| **Global Standard Deviation ($\sigma_{\text{global}}$)** | Spread of individual predictions around the mean. | Stability: lower standard deviation reflects consistent behavior across quadruplets. |
| **Category Means ($\overline{S}_1, \dots, \overline{S}_{14}$)** | Mean alignment computed separately for each of the 14 categories. | Granular diagnostic: isolates category-specific strengths and weaknesses. |
| **Category Standard Deviation ($\sigma_{\text{categories}}$)** | Standard deviation across the 14 category means. | Domain consistency: low values indicate balanced capability; high values indicate narrow specialization. |
| **Semantic vs. Syntactic Means** | Mean performance on semantic categories contrasted with syntactic categories. | Compares whether an architecture favors topical semantics or morphological syntax. |
| **Cosine Distance ($\overline{D}_{\text{global}}, \sigma_{\text{global}}$)** | Original assignment metric in $[0, 2]$ where lower is better. | Preserved for strict compliance with the assignment specification. |

---

## 2. Computational Resource Consumption and Hardware Profiling

To evaluate the trade-off between representation quality and compute cost (Pareto frontier), each model records:
- **Hardware Profile:** CPU model, logical core count, total physical RAM, and accelerator profile (when running in Colab).
- **Parallelism:** Active worker threads (`NUM_WORKERS = max(1, cpu_count)`).
- **Execution Times:** Training duration (`train_time_sec`), evaluation duration (`eval_time_sec`), and step elapsed duration.
- **CPU Time:** Cumulative processor execution time (`cpu_time_sec`).
- **Memory Footprint:** Peak resident set memory (`peak_ram_mb`).
- **Processing Throughput:** Thousands of words processed per second (`throughput_kwords_sec`).

---

## 3. Resilience and Checkpoint Recovery

1. **Immediate Disk Checkpointing (`TP1/outputs/experiment_results.csv`):**
   * As soon as model $k$ completes training and evaluation, its record (hyperparameters, evaluation metrics, hardware telemetry) is committed to disk in CSV format.
   * Results never reside exclusively in volatile memory.
2. **Automatic Resume Capability:**
   * Prior to launching training for model $k$, the runner checks whether its identifier already exists in the CSV file.
   * If recorded, it skips execution immediately and advances to the next configuration.
   * If a session disconnects midway through the grid, re-executing the cell resumes precisely from the uncompleted configuration.
3. **Progress Telemetry and Execution Logs:**
   * Structured console logs report step index `[k/54]`, iteration elapsed time, peak RAM, and dynamic estimated completion time (ETA).

---

## 4. Phased Implementation Roadmap

### Phase 1: Modular Training Function (Section 3) - COMPLETED
* Implementation of `train_word2vec_model(...)` with deterministic seed (`seed=42`) and CPU multi-worker streaming.
* Smoke test validation completed in 53 seconds.

### Phase 2: Vector Evaluation Engine (Section 4) - COMPLETED
* Implementation of `evaluate_word2vec_analogies(model, df_analogies)` supporting both Cosine Distance $[0, 2]$ and Alignment Score $[-1, 1]$.
* Smoke model verification achieved 91.21% analogy coverage over vocabulary.

### Phase 3: Controlled 54-Model Grid Search (Section 5) - COMPLETED
* Parameter search vectors (`GRID_ARCHITECTURES`, `GRID_WINDOWS`, `GRID_VECTOR_SIZES`, `GRID_EPOCHS`).
* Hardware resource monitoring (CPU, RAM, throughput, wall-clock time).
* Incremental checkpointing to `TP1/outputs/experiment_results.csv`.
* **Status:** All 54 models executed successfully (100% completion). Champion model identified: `M28_SG_w2_d50_e1` (Alignment: +0.5171, Global Mean Cosine Distance: 0.4829, Training Duration: 21.65s).

### Phase 4: Scientific Visualizations and Conclusion (Sections 6 and 7) - COMPLETED
* **Section 6: Comparative Empirical Visualizations (100% English, Individual Standalone Figures):**
  1. *Cell 6.1 (Global Performance Spectrum):* Standalone S-curve ranking all 54 models from Rank 1 to 54 against Cosine Distance, highlighting champion M28 at Rank #1 with a gold star.
  2. *Cell 6.2 (Top 10 Configurations Leaderboard):* Standalone horizontal bar chart with bold internal annotations displaying metrics for the top 10 models.
  3. *Cell 6.3 (Category Cosine Distance Distributions):* Standalone horizontal boxplots across all 14 categories with explicit mean diamonds ($\mu$), median lines ($Md$), and M28 markers.
  4. *Cell 6.4 (Macro Section Comparison):* Standalone boxplot comparing Semantic vs. Syntactic section error distributions.
  5. *Cell 6.5 (Context Window Dynamics):* Standalone line plot tracing window size impact ($w \in \{2, 5, 10\}$) by architecture with standard deviation bands.
  6. *Cell 6.6 (Dimension vs. Epochs Heatmap):* Standalone Skip-gram matrix heatmap crossing vector size and training epochs.
  7. *Cell 6.7 (Dual-Axis Trade-off Balance Curve):* Standalone dual-curve figure showing Cosine Distance descending while Training Time ascends, highlighting the M28 Optimal Balance Point.
  8. *Cell 6.8 (Pareto Efficiency Frontier):* Standalone scatter plot of latency vs. distance, bubble size scaled by peak RAM (MB), tracing the Pareto optimal frontier.
* **Section 7: Results Discussion (Pure Markdown):**
  - High-signal synthesis presenting the best-performing model (`M28_SG_w2_d50_e1`) and its metrics.
  - Core empirical takeaways from Section 6 plots (spectrum, category distributions, trade-offs).
  - Hyperparameter sensitivity overview analyzing Architecture, Context Window, Dimension, and Epochs.
