# Technical and Scientific Specification (spec.md)

**Project:** Practical Assignment 1 (TP1) - Deep Learning for Natural Language Processing  
**Subject:** Word Embeddings Hyperparameter Exploration and Vector Analogy Evaluation  
**Reference Document:** [Original Assignment PDF](./enunciado.pdf)

---

## 1. Scientific Context and Objective

Distributed word representations (*Word Embeddings*) map discrete vocabulary tokens $w \in V$ into dense, continuous vectors in a lower-dimensional latent space $\mathbb{R}^d$ ($d \ll |V|$). Unlike sparse, high-dimensional co-occurrence matrices or one-hot encodings, distributed embeddings capture semantic and syntactic nuances based on the **distributional hypothesis** (*Harris, 1954; Firth, 1957*): *"You shall know a word by the company it keeps"*.

The objective of this assignment is to systematically analyze how key hyperparameter choices influence the topology, linear regularity, and general representation quality of word vectors produced by the Word2Vec model, using an intrinsic evaluation based on **word analogies**.

---

## 2. Data and Resource Requirements

### 2.1. Training Corpus
- **Dataset:** `text8`
- **Source URL:** `http://mattmahoney.net/dc/text8.zip`
- **Description:** First 100,000,000 characters from an English Wikipedia dump, cleaned of punctuation and non-alphabetic characters (leaving only lower-case letters `a-z` and spaces).
- **Scale:** ~100 MB uncompressed text (~17 million word tokens).

### 2.2. Evaluation Dataset
- **Dataset:** `questions-words.txt` (Google Analogy Test Set; Mikolov et al., 2013)
- **Source URL:** `https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt`
- **Format:** Structured in topical categories (e.g., `: capital-common-countries`, `: past-tense`). Each test instance consists of four words: $w_1, w_2, w_3, w_4$.
  - Example: `Paris France Berlin Germany` represents the proportional relationship:
    $$\text{Paris} : \text{France} :: \text{Berlin} : \text{Germany}$$

---

## 3. Hyperparameter Space

The assignment mandates the exploration and comparison of the following four hyperparameter axes:

| Hyperparameter | Parameter Name (`gensim`) | Description | Target Evaluation Values |
| :--- | :--- | :--- | :--- |
| **Architecture** | `sg` | `0` for **CBOW** (Continuous Bag of Words); `1` for **Skip-gram** | `{CBOW, Skip-gram}` |
| **Context Window Size** | `window` | Maximum symmetric distance between target and context tokens | `{2, 5, 10}` |
| **Embedding Dimension** | `vector_size` | Dimensionality of the latent continuous space $d$ | `{50, 100, 300}` |
| **Training Epochs** | `epochs` | Number of iterations/sweeps over the entire corpus | `{1, 5, 10}` |

### 3.1. Theoretical Foundations of Evaluated Parameters
- **CBOW vs. Skip-gram:**
  - *CBOW (`sg=0`)* predicts the center target token $w_t$ given the average vector of its context tokens. It is computationally faster and often excels at learning syntactic patterns from frequent words.
  - *Skip-gram (`sg=1`)* predicts surrounding context words given the center word $w_t$. It creates more training pairs per token, exhibiting superior representations for infrequent words and nuanced semantic relations.
- **Context Window (`window`):**
  - Smaller windows ($2$ to $3$) emphasize local syntactic constraints (part-of-speech, immediate modifiers).
  - Larger windows ($5$ to $10$) capture broader topical, associative, and semantic relationships.
- **Embedding Dimension (`vector_size`):**
  - Lower dimensions ($50$) reduce model capacity, risking underfitting.
  - Higher dimensions ($300$) allow richer geometric disentanglement, but require larger corpora and higher compute budgets to avoid overfitting/sparsity.
- **Training Epochs (`epochs`):**
  - Dictates optimization convergence vs. computational training time and potential saturation.
- **Vocabulary Threshold (`min_count = 5`, fixed baseline):**
  - Inherited from Mikolov et al. (2013) and Gensim defaults. Words appearing fewer than 5 times in the corpus (~17M tokens) lack sufficient co-occurrence contexts for stochastic gradient updates to stabilize vector coordinates. Pruning them eliminates isolated typographical noise and bounds the vocabulary matrix memory, while true *hapax legomena* ($f=1$) and rare tokens ($1 < f < 5$) are tracked in frequency profiling.

---

## 4. Evaluation Methodology via Vector Algebra

Each trained model is evaluated using algebraic vector transformations over the test quadruplets $w_1, w_2, w_3, w_4$.

### 4.1. Algebraic Analogy Formulation
For an analogy relationship $w_1 : w_2 :: w_3 : w_4$ (e.g., `Paris France Berlin Germany`, where $w_1 = \text{Paris}, w_2 = \text{France}, w_3 = \text{Berlin}, w_4 = \text{Germany}$), the assignment sheet mandates the literal vector algebraic transformation:
$$\vec{R} = \vec{v}(w_2) + \vec{v}(w_1) - \vec{v}(w_3) = \vec{v}(\text{France}) + \vec{v}(\text{Paris}) - \vec{v}(\text{Berlin})$$
The resultant vector $\vec{R}$ is the predicted geometric coordinate for the target word $w_4$ ($\text{Germany}$).

### 4.2. Metric: Cosine Distance and Normalized Alignment Score
For each individual analogy quadruplet $i$, the cosine similarity between the resultant vector $\vec{R}_i$ and the expected target word vector $\vec{v}(w_{4, i})$ is computed as:
$$\text{sim}_{\cos}(\vec{R}_i, \vec{v}(w_{4, i})) = \frac{\vec{R}_i \cdot \vec{v}(w_{4, i})}{\|\vec{R}_i\|_2 \|\vec{v}(w_{4, i})\|_2}$$

1. **Cosine Distance ($d_i \in [0, 2]$):** Formally mandated by the assignment sheet:
   $$d_i = \text{dist}_{\cos}(\vec{R}_i, \vec{v}(w_{4, i})) = 1 - \text{sim}_{\cos}(\vec{R}_i, \vec{v}(w_{4, i})) \quad \in [0, 2]$$
   Where $d_i = 0$ indicates perfect alignment, and values closer to $0$ represent superior performance.

2. **Intuitive Alignment Score ($s_i \in [-1, 1]$):** To maximize human interpretability across visualizations and analysis, the cosine distance is mapped linearly to:
   $$s_i = 1 - d_i = \text{sim}_{\cos}(\vec{R}_i, \vec{v}(w_{4, i})) \quad \in [-1, 1]$$
   Where $s_i = +1.0$ indicates perfect alignment ($0^\circ$), $s_i = 0.0$ indicates orthogonality ($90^\circ$), and $s_i = -1.0$ indicates diametric opposition ($180^\circ$). All comparative charts, heatmaps, and visualizations follow this $[-1, 1]$ alignment scale (higher is better).

### 4.3. Statistical Aggregation and Dispersion Metrics
To provide rigorous statistical characterization beyond simple averages, the evaluation pipeline computes:

1. **Global Alignment Mean, Distance Mean, and Standard Deviation:**
   - Over all $N$ valid quadruplets in the benchmark:
     $$\overline{S}_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} s_i, \qquad \overline{D}_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} d_i = 1 - \overline{S}_{\text{global}}$$
     $$\sigma_{\text{global}} = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} (s_i - \overline{S}_{\text{global}})^2} = \sigma(d)$$
   - Quantifies overall representation quality and variance.

2. **Stratified Mean and Standard Deviation across 14 Benchmark Categories:**
   - For each category $c \in \{1, \dots, 14\}$ with $N_c$ quadruplets:
     $$\overline{S}_c = \frac{1}{N_c} \sum_{i \in c} s_i, \qquad \sigma_c = \sqrt{\frac{1}{N_c - 1} \sum_{i \in c} (s_i - \overline{S}_c)^2}$$
   - Identifies specific linguistic strengths and weaknesses (e.g., capitals vs. verb inflections).

3. **Section-Level Aggregation (Semantic vs. Syntactic Blocks):**
   - Computes separate $(\overline{S}_{\text{sem}}, \sigma_{\text{sem}})$ across semantic categories and $(\overline{S}_{\text{syn}}, \sigma_{\text{syn}})$ across syntactic categories.

4. **Inter-Category Consistency Metric:**
   - The standard deviation of the 14 category means $\text{std}(\{\overline{S}_1, \dots, \overline{S}_{14}\})$ is tracked to quantify how evenly the model performs across diverse semantic and grammatical tasks.

- **Primary Decision Rule:** The optimal hyperparameter combination maximizes the global mean alignment $\overline{S}_{\text{global}}$ (equivalently minimizing $\overline{D}_{\text{global}}$), with lower standard deviation $\sigma_{\text{global}}$ serving as the tie-breaker criterion for stability.

---

## 5. Google Colab & Remote Runtime Adaptations

To ensure seamless execution when connected to a Google Colab kernel (with or without a T4 GPU runtime), the codebase and notebook adhere to the following specifications:

### 5.1. Mandatory Execution via Google Colab Kernel

> **MANDATORY RUNTIME DIRECTIVE:**  
> All notebook executions, both by the user/student within the IDE and by the AI assistant, **must be performed strictly through the connected Google Colab remote kernel**.  
> Neither the student in the IDE nor the AI assistant may execute notebook cells using a local environment or local kernel.

- **Mandatory Target Runtime:** Google Colab Remote Kernel (connected to the IDE via the remote Jupyter server URI).
- **Division of Execution Responsibilities:**
  - **User/Student Execution:** The student executes interactive cells, full corpus downloads, multi-epoch Word2Vec training loops, and the complete hyperparameter grid sweep directly within the IDE using the connected Colab kernel.
  - **AI Assistant Execution & Validation:** The AI assistant configures all dependencies, commands, and logic specifically for the Colab runtime container, and performs fast compilation and syntax validation without running long-blocking commands that create operational bottlenecks.
- **Environment Parity:** All dependencies, file path resolutions (`/content` vs. relative paths), and execution cells must be configured to run natively inside the Colab runtime container.

### 5.2. Colab Shell Syntax for Package Management
- All pip installation commands within notebook code cells must strictly begin with an exclamation mark (`!`) rather than `%`:
  ```bash
  !pip install -q gensim numpy scipy matplotlib seaborn pandas
  ```
- Flags such as `-q` (quiet) must be included to avoid cluttering notebook execution outputs.

### 5.3. Dynamic Environment and Path Resolution
- **Runtime Detection:** The execution pipeline must dynamically detect whether it is executing inside a Google Colab virtual machine:
  ```python
  IS_COLAB = "google.colab" in sys.modules
  ```
- **Working Directories:** 
  - In Colab, the primary directory defaults to `/content/TP1` or `/content`.
  - In local environments, paths resolve relative to the repository (`./data` or `../data`).
  - The notebook must automatically construct and verify local data paths (`Path(DATA_DIR).mkdir(parents=True, exist_ok=True)`).

### 5.4. Automated Dataset Acquisition
- Because Colab instances operate in ephemeral containers, datasets cannot be assumed to pre-exist on disk.
- The pipeline must include self-healing, automated download logic using standard libraries (`urllib.request`) to fetch:
  1. `text8.zip` from `http://mattmahoney.net/dc/text8.zip`, extract `text8`, and immediately remove `text8.zip` to prevent keeping duplicate archives.
  2. `questions-words.txt` from `https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt`.
- Existing files must be verified before downloading to skip redundant network requests and avoid file duplication.

### 5.5. Hardware & Resource Optimization in Colab
- **Worker Threads:** Word2Vec training parallelization should dynamically adapt to available CPU cores:
  ```python
  NUM_WORKERS = min(4, os.cpu_count() or 2)
  ```
- **Memory Streaming:** Use `gensim.models.word2vec.Text8Corpus` for generator-based disk streaming, preventing memory spikes that could trigger Colab's 12 GB RAM kill threshold.
- **Hardware Logging:** When running on a Colab GPU instance (e.g. NVIDIA T4), detect and log GPU specifications (`!nvidia-smi`), while maintaining documented theoretical explanations that Gensim's Word2Vec optimizes on multi-threaded CPU instructions (SIMD/BLAS).

### 5.6. Persistent Artifact Logging & Disconnect-Resilient Checkpointing
- **Output Directory:** All intermediate checkpoints, model performance logs, detailed metric tables, and final evaluation outputs must be saved directly to `TP1/outputs/` (which is excluded from git tracking).
- **Incremental Checkpointing:** After evaluating each individual model configuration $k \in \{1, \dots, 54\}$, append the detailed results row immediately to `TP1/outputs/experiment_results.csv`.
- **Fault-Tolerant Resume Capability:** To safeguard against Google Colab idle session disconnects or runtime timeouts during the 54-model grid run:
  - The grid runner inspects `TP1/outputs/experiment_results.csv` upon initialization.
  - Any configuration that was already trained, evaluated, and saved is skipped automatically.
  - If a session is interrupted at model $k$, re-executing the cell immediately resumes from model $k+1$ without recalculating previous models.

---

### 5.7. Computational Resource and Hardware Profiling
To evaluate the trade-off between semantic representation quality and computational resource consumption (Pareto efficiency frontier):
- **Hardware Profile Logging:** Each model execution records the underlying platform environment: CPU model name, physical and logical core counts, available RAM (GB), GPU device identifier (if active), and the number of parallel workers deployed (`workers = max(1, os.cpu_count())`).
- **Resource Metrics Recorded:**
  - `train_time_sec`: Wall-clock training duration.
  - `eval_time_sec`: Vectorized analogy evaluation runtime.
  - `total_time_sec`: Complete step duration.
  - `cpu_time_sec`: Accumulated process CPU time across all threads.
  - `peak_ram_mb`: Resident Set Size (RSS) peak memory footprint.
  - `throughput_kwords_sec`: Effective token processing throughput.
- **Empirical Trade-off Analysis:** In downstream analyses, configurations are compared along both performance (mean alignment score $s \in [-1, 1]$) and resource efficiency dimensions to identify the optimal cost-benefit threshold.

## 6. Deliverable Requirements

The final deliverable is an end-to-end executable **Jupyter Notebook (`.ipynb`)** satisfying the following standards:
1. **Self-Contained & Reproducible:** Includes all imports, automated data acquisition, training loops, evaluation metrics, and plots.
2. **Notebook Cell Standards:**
   - First line of every code cell must follow the mandatory format: `#CELL S.C - BRIEF DESCRIPTION IN 1 SENTENCE`.
   - Clear markdown sections and subsections preceding code cells.
   - Centralized imports and dependency installations in top initial cells.
   - Strictly no references to internal repository specifications (`spec.md`, `agents.md`).
3. **Visualization & Scientific Reporting:**
   - Consolidated summary tables of all tested configurations.
   - Comparative charts (heatmaps, line plots of window/dimension effects, error bars).
   - Low-dimensional 2D visualization (t-SNE or PCA) displaying semantic clusters in the learned space.
   - Explicit scientific conclusions explaining the best configuration and theoretical justifications.
4. **Concise, Direct, and Non-Redundant Narrative:**
   - Markdown cells must be direct, high-signal, and strictly non-repetitive across the notebook.
   - Eliminate redundant recaps of corpus metadata, repetitive parameter descriptions, and circular theoretical explanations. Each section must cover only its specific domain logic and concepts concisely.
