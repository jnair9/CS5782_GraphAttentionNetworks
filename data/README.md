## Dataset

We use the Protein-Protein Interaction (PPI) dataset from the original Graph Attention Networks paper.
The dataset is automatically downloaded when running the notebook.
The dataset is loaded using PyTorch Geometric:

```python
from torch_geometric.datasets import PPI

train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')