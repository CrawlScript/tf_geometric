# coding=utf-8

# https://github.com/KarolisMart/DropGNN/blob/main/gin-synthetic.py

from tf_geometric.data.dataset import Dataset 
import numpy as np
import networkx as nx

def _compute_degree(edge_index, num_nodes):
    # use numpy.add.at
    row, col = edge_index
    degree = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(degree, row, 1)
    return degree



def _create_ports(edge_index, num_nodes):
    row, col = edge_index
    degree = _compute_degree(edge_index, num_nodes)
    ports = np.zeros(edge_index.shape[1])

    for node_index in range(num_nodes):
        node_ports = np.random.permutation(degree[node_index])
        for neighbor_index, neighbor_node_index in enumerate(col[row == node_index]):
            ports[np.logical_and(row == node_index, col == neighbor_node_index)] = node_ports[neighbor_index]

    return ports

def _create_x(num_nodes):
    return np.ones((num_nodes, 1))

def _create_id(num_nodes):
    return np.random.permutation(num_nodes)



class LimitsOneDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def load_data(self):
        num_nodes = 16 # There are two connected components, each with 8 nodes
        
        ports = [1,1,2,2] * 8
        colors = [0, 1, 2, 3] * 4

        y = np.array([0]* 8 + [1] * 8)
        edge_index = np.array([
            [0,1,1,2, 2,3,3,0, 4,5,5,6, 6,7,7,4, 8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,8], 
            [1,0,2,1, 3,2,0,3, 5,4,6,5, 7,6,4,7, 9,8,10,9,11,10,12,11,13,12,14,13,15,14,8,15]
        ])

        x = np.zeros([num_nodes, 4])
        
        x[range(num_nodes), colors] = 1

        node_ids =np.random.permutation(np.arange(num_nodes))

        return x, edge_index, y, node_ids, ports


class LimitsTwoDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def load_data(self):
        num_nodes = 16 # There are two connected components, each with 8 nodes

        ports = ([1,1,2,2,1,1,2,2] * 2 + [3,3,3,3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = np.array([0] * 8 + [1] * 8)
        edge_index = np.array([
            [0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 1,3,5,7, 8,9,9,10,10,11,11,8, 12,13,13,14,14,15,15,12, 9,15,11,13], 
            [1,0,2,1,3,2,0,3, 5,4,6,5,7,6,4,7, 3,1,7,5, 9,8,10,9,11,10,8,11, 13,12,14,13,15,14,12,15, 15,9,13,11]]
        )
        x = np.zeros((num_nodes, 4))
        x[range(num_nodes), colors] = 1

        node_ids =np.random.permutation(np.arange(num_nodes))
        return x, edge_index, y, node_ids, ports




class LCCDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False

    def load_data(self):
        generated = False

        try_count = 0

        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)

                if nx.is_connected(nx_g):
                    i += 1
                    
                    nx_g = nx_g.to_directed()
                    edge_index = np.array(nx_g.edges).T

                    # data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [int(nb) for nb in edge_index[1][edge_index[0]==n]]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if np.logical_and(edge_index[0]==nb1, edge_index[1]==nb2).any():
                                    edges += 1
                        lbls[n] = int(edges/2)
                    y = np.array(lbls)
                    labels.extend(lbls)

                    ports = _create_ports(edge_index, size)
                    x = _create_x(size) 
                    node_ids = _create_id(size)

                    graph = {
                        "x": x,
                        "edge_index": edge_index,
                        "y": y,
                        "ports": ports,
                        "node_ids": node_ids
                    }

                    graphs.append(graph)

            # print("label 0:", labels.count(0), "label 1:", labels.count(1), "label 2:", labels.count(2))
                
            generated = labels.count(0) >= 10 and labels.count(1) >= 10 and labels.count(2) >= 10 # Ensure the dataset is somewhat balanced

        return graphs
        



class TrianglesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False

    def load_data(self):
        size = self.num_nodes
        generated = False
        count = 0
        while not generated:
            count += 1

            if count % 1000 == 0:
                print("count:", count)

            nx_g = nx.random_degree_sequence_graph([3] * size)
            nx_g = nx_g.to_directed()
            edge_index = np.array(nx_g.edges).T

            labels = [0] * size
            for n in range(size):
                for nb1 in edge_index[1][edge_index[0]==n]:
                    for nb2 in edge_index[1][edge_index[0]==n]:
                        if np.logical_and(edge_index[0]==nb1, edge_index[1]==nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        y = np.array(labels)

        ports = _create_ports(edge_index, size)
        x = _create_x(size) 
        node_ids = _create_id(size)

        return x, edge_index, y, node_ids, ports






# graphs = LCC().load_data()
# for graph in graphs:
#     print(graph)
        
# x, edge_index, y, node_ids, ports = LimitsOneDataset().load_data()

# print("x:", x)
# print("edge_index:", edge_index)
# print("y:", y)
# print("node_ids:", node_ids)
# print("ports:", ports)
