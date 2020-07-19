# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg

source_index = [0, 0, 1, 1, 1, 2, 2]
score = [0.2, 0.4, 0.1, 0.3, 0.5, 0.4, 0.1]

topk_node_index = tfg.nn.topk_pool(source_index, score, k=1)
print(topk_node_index)


source_index = [1, 0, 0, 2, 1, 2, 1]
score = [0.2, 0.4, 0.1, 0.3, 0.5, 0.4, 0.1]

topk_node_index = tfg.nn.topk_pool(source_index, score, k=1)
print(topk_node_index)
