# Deep Learning for Natural Language Processing: Practical Assignments

Repository hosting the practical assignments for the Deep Learning for NLP course at the **Universidade Federal de Minas Gerais (UFMG)**.

---

## 📁 Repository Structure

```text
TPs-Deep-Learning-for-NLP/
├── README.md               # Repository documentation (this file)
└── TP1/                    # Practical Assignment 1: Word Embeddings (Word2Vec)
    └── src/                # Implementation notebook
        └── word2vec_hyperparameters.ipynb
```

---

## 🚀 Quick Setup

### Local Environment
```bash
# Clone and enter the repository
git clone https://github.com/gustavofcunha/TPs-Deep-Learning-for-NLP.git
cd TPs-Deep-Learning-for-NLP

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install gensim numpy scipy matplotlib seaborn jupyter ipykernel
```

### Google Colab
If executing on Google Colab, the notebook automatically installs the required dependencies in its initial cell via `!pip install -q gensim numpy scipy matplotlib seaborn`.

---

## 🔬 Practical Assignment 1 (TP1): Word Embeddings (Word2Vec)

### Overview
This assignment investigates dense distributed representations (**word embeddings**) trained with **Word2Vec** on the `text8` corpus (~17M tokens). We systematically analyze how hyperparameter configurations influence the geometric quality of the resulting vector space.

### Hyperparameters Explored
- **Architecture:** CBOW (`sg=0`) vs. Skip-gram (`sg=1`)
- **Context Window Size:** `2`, `5`, and `10`
- **Embedding Dimension:** `50`, `100`, and `300`
- **Training Epochs:** `1`, `5`, and `10`

### Evaluation
Models are evaluated on the **Google Analogy Test Set** (`questions-words.txt`, ~19.5k quadruplets $A : B :: C : D$) using the vector algebraic operation specified in the course assignment:
$$\vec{R} = \vec{v}(B) + \vec{v}(A) - \vec{v}(C)$$

Model quality is quantified using two complementary metrics:
1. **Cosine Distance ($d \in [0, 2]$):** The formal loss metric from the assignment handout, where lower values indicate closer proximity to the expected target vector $\vec{v}(D)$ ($d=0$ is an exact match).
2. **Intuitive Alignment Score ($s = 1 - d \in [-1, 1]$):** A normalized directional alignment score where $+1$ represents perfect alignment ($0^\circ$), $0$ represents orthogonality ($90^\circ$), and $-1$ represents diametric opposition ($180^\circ$). All comparative charts, heatmaps, and ranking tables follow this intuitive $[-1, 1]$ scale (higher is better).

For each configuration, the pipeline computes the **Global Mean and Standard Deviation**, as well as stratified metrics across all 14 individual categories and semantic vs. syntactic blocks. 

### Resource Profiling and Pareto Efficiency Trade-offs
To evaluate the trade-off between representation quality and computational cost, the pipeline profiles:
- **Hardware Context:** Platform CPU model, core counts, RAM availability, and GPU device.
- **Resource Utilization:** Wall-clock training duration (`train_time_sec`), evaluation time, accumulated CPU time, peak memory footprint (`peak_ram_mb`), and processing throughput (`throughput_kwords_sec`).
- **Parallelism:** Multiprocessing across CPU cores (`workers = os.cpu_count()`).

### Experiment Checkpointing and Resilience
The 54 grid combinations ($2 \times 3 \times 3 \times 3$) are executed with persistent checkpointing: each model's evaluation metrics and resource statistics are appended immediately to `TP1/outputs/experiment_results.csv`. If a Google Colab session disconnects, the execution script automatically detects completed models and resumes from where it stopped without duplicating compute.

### Running the Notebook
Open and execute the self-contained notebook:
```bash
jupyter notebook TP1/src/word2vec_hyperparameters.ipynb
```