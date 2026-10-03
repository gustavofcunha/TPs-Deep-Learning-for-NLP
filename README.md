# Deep Learning for Natural Language Processing — Practical Assignments

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
Models are evaluated on the **Google Analogy Test Set** (`questions-words.txt`, ~19k quadruplets $A : B :: C : D$) using algebraic vector composition:
$$\vec{R} = \vec{v}(B) - \vec{v}(A) + \vec{v}(C)$$

The performance is quantified by the **Mean Cosine Distance** between the predicted vector $\vec{R}$ and the expected target vector $\vec{v}(D)$. The optimal model configuration is the one that minimizes this distance.

### Running the Notebook
Open and execute the self-contained notebook:
```bash
jupyter notebook TP1/src/word2vec_hyperparameters.ipynb
```