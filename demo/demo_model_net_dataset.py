# coding=utf-8
from tf_geometric.datasets import ModelNet10Dataset, ModelNet40Dataset

train_graphs, test_graphs, label_names = ModelNet10Dataset().load_data()

for graph in test_graphs:
    print(graph.y)


