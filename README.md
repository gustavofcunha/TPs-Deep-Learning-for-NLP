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
pip install gensim numpy scipy matplotlib seaborn pandas jupyter ipykernel
```

### Google Colab
If executing on Google Colab, the notebook automatically installs the required dependencies in its initial cell via `%pip install -q gensim numpy scipy matplotlib seaborn pandas`.

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
1. **Cosine Distance ($d \in [0, 2]$):** The formal metric from the assignment handout, where lower values indicate closer proximity to the expected target vector $\vec{v}(D)$ ($d=0$ is an exact match).
2. **Intuitive Alignment Score ($s = 1 - d \in [-1, 1]$):** A normalized directional alignment score where $+1$ represents perfect alignment ($0^\circ$), $0$ represents orthogonality ($90^\circ$), and $-1$ represents diametric opposition ($180^\circ$).

### Empirical Findings and Winning Configuration
Across all 54 systematically evaluated models, configuration **`M28_SG_w2_d50_e1`** achieved the global minimum cosine distance:
- **Architecture:** Skip-gram (`sg=1`)
- **Window:** $w = 2$
- **Dimension:** $d = 50$
- **Epochs:** $e = 1$
- **Global Mean Cosine Distance:** **0.4829** (Global Alignment: **+0.5171**)
- **Training Duration:** **21.65 seconds** (85.5% faster than experimental average)

### Running the Notebook
Open and execute the self-contained notebook:
```bash
jupyter notebook TP1/src/word2vec_hyperparameters.ipynb
```