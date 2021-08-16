# coding=utf-8
from tf_geometric.datasets import ModelNet10, ModelNet40

train_graphs, test_graphs, label_names = ModelNet40().load_data()

for graph in test_graphs:
    print(graph.y)


