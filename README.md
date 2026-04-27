# Graph Attention Networks Reimplementation (PPI)

## 1. Introduction

This repository contains our reimplementation of **Graph Attention Networks** by Veličković et al. (ICLR 2018) on the inductive Protein-Protein Interaction (PPI) task.

Graph Attention Networks (GATs) are graph neural networks that use a learned attention mechanism to assign different importance weights to each neighboring node during feature aggregation. This lets the model focus on more informative neighbors instead of treating every connection equally.

We compare learned attention against a **Const-GAT** baseline, where all neighbors receive fixed uniform attention weights.

---

## 2. Chosen Result

We reproduce the **inductive node classification experiment on the PPI dataset**, corresponding to the PPI result table from the original GAT paper.

In this benchmark, the model is trained on several graphs and evaluated on unseen test graphs. This tests whether the model can generalize to new graph structures rather than memorize one fixed graph.

The task is multi-label classification, where each protein may belong to multiple functional categories, so performance is measured using **micro-averaged F1 score**.

The original paper reports:

| Method | PPI Micro-F1 |
|---|---:|
| GraphSAGE* | 0.768 |
| Const-GAT | 0.934 ± 0.006 |
| GAT | 0.973 ± 0.002 |

Our goal was to reproduce the GAT result within approximately ±0.01 of the reported score while preserving the performance gap between GAT and the non-attention Const-GAT baseline.

---

## 3. GitHub Contents

| Folder/File | Description |
|---|---|
| `code/` | Google Colab notebook containing the implementation, training loop, and evaluation code |
| `data/` | README explaining how the PPI dataset is loaded |
| `results/` | Generated plots, result tables, and training outputs |
| `poster/` | Final project poster PDF |
| `report/` | Final 2-page report PDF |
| `README.md` | Overview of the project, implementation, reproduction steps, and results |

---

## 4. Re-implementation Details

We implemented GAT using **PyTorch Geometric** in Google Colab instead of the original TensorFlow implementation. PyTorch Geometric provides standard graph neural network tooling and built-in access to the PPI dataset.

The core model is a graph attentional layer where each node aggregates features from its first-order neighbors using learned, normalized attention coefficients.

For each directed edge \((i, j)\), the raw attention score is computed as:

```math
e_{ij} = \text{LeakyReLU}(a^T [W h_i \, || \, W h_j])
```

where \(W\) is a shared linear transformation, \(a\) is a learnable attention vector, and \(||\) denotes concatenation.

These scores are normalized using softmax over each node's neighborhood and then used to compute a weighted sum of neighbor features.

### Architecture

Our GAT implementation uses a 3-layer architecture:

- First layer: 4 attention heads, each computing 256 features
- Second layer: 4 attention heads, each computing 256 features
- Final layer: 6 attention heads, each computing 121 output features
- ELU activations after hidden layers
- Logistic sigmoid output for multi-label classification
- Residual / skip connections added to improve optimization and stability

We also implemented **Const-GAT**, which removes learned attention by assigning uniform attention weights to all neighbors. This acts like a GCN-style inductive operator and lets us test how much learned attention contributes beyond neighborhood aggregation.

### Training Setup

We trained using:

- Optimizer: Adam
- Learning rate: 0.005
- Batch size: 2 graphs
- Early stopping: patience of 100 epochs based on validation micro-F1
- Gradient clipping to reduce occasional training instability from exploding gradients
- Evaluation metric: micro-averaged F1 score
- 10 independent runs for each model

Training was run on Google Colab using a T4 GPU. Each run took roughly 20 minutes, and the full set of 20 runs took about 7 hours total.

---

## 5. Reproduction Steps

### 1. Clone the repository

```bash
git clone https://github.com/jnair9/CS5782_GraphAttentionNetworks.git
cd CS5782_GraphAttentionNetworks
```

### 2. Install dependencies

The easiest way to reproduce the project is to run the notebook in Google Colab.

Required libraries include:

```bash
pip install torch torch-geometric scikit-learn matplotlib pandas numpy
```

If running locally, make sure your PyTorch and PyTorch Geometric versions are compatible with your CUDA version.

### 3. Open the notebook

Open the notebook:

```text
code/GAT_PPI_Reimplementation.ipynb
```

Then run all cells.

### 4. Dataset loading

The PPI dataset is loaded directly through PyTorch Geometric:

```python
from torch_geometric.datasets import PPI

train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')
```

The dataset is automatically downloaded into `data/PPI/` on the first run. No manual dataset download is required.

### 5. Expected compute

A GPU is strongly recommended. We used Google Colab with a T4 GPU.

Approximate runtime:

- ~20 minutes per training run
- 10 runs for GAT-Res
- 10 runs for Const-GAT-Res
- ~7 hours total for the full experiment

---

## 6. Results / Insights

Our implementation produced the following 10-run results on the PPI test set:

| Method | PPI Test F1 |
|---|---:|
| Random | 0.396 |
| MLP | 0.422 |
| GraphSAGE* | 0.768 |
| Const-GAT (paper) | 0.934 ± 0.006 |
| GAT (paper) | 0.973 ± 0.002 |
| Const-GAT-Res (ours) | 0.9574 ± 0.0004 |
| GAT-Res (ours) | 0.9719 ± 0.0024 |

GAT-Res nearly matches the original GAT paper result, reaching **0.9719 ± 0.0024** compared to the paper's **0.973 ± 0.002**.

Const-GAT-Res outperforms the paper's Const-GAT baseline, reaching **0.9574 ± 0.0004** compared to **0.934 ± 0.006**. This improvement is likely due to our residual connections and modern training setup.

### Main observations

- **Learned attention improves performance:** GAT-Res achieves higher F1 than Const-GAT-Res, showing that adaptive neighbor weighting matters.
- **Const-GAT-Res is more stable:** Const-GAT-Res has very low variance across runs because it removes the learnable attention mechanism, making training less sensitive to initialization and stochasticity.
- **Residual connections help optimization:** Residual connections improved stability and helped both models train reliably.
- **Our GAT-Res closely reproduces the paper:** The final GAT-Res score is within the target range of the original reported GAT result.

Overall, the results suggest that attention provides the main performance benefit, while residual connections improve training stability and reproducibility.

---

## 7. Conclusion

Our reimplementation confirms that Graph Attention Networks achieve strong performance on inductive graph learning tasks. The learned attention mechanism improves performance by allowing the model to weight neighbors differently, while residual connections improve optimization and stability.

GAT-Res achieved near paper-level performance on PPI, and Const-GAT-Res showed that even fixed attention can perform strongly when combined with residual connections. Together, these results support the original paper's claim that attention is useful for graph representation learning.

---

## 8. References

- Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. **Graph Attention Networks.** International Conference on Learning Representations (ICLR), 2018. https://arxiv.org/abs/1710.10903
- Brody, Shaked, Uri Alon, and Eran Yahav. **How Attentive Are Graph Attention Networks?** International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2105.14491
- PyTorch Geometric Documentation. https://pytorch-geometric.readthedocs.io/

---

## 9. Acknowledgements

This project was completed as part of **CS 4782 / CS 5782 Deep Learning** at Cornell University.

Project members:

- Anish Bhupalam
- Anagha A. Ram
- Jason Nair
