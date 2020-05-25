fp = '/datasets/dsc180a-wi20-public/Malware/group_data/group_02/sensitive_data/interim/graph/graph.mat'

import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
import pandas as pd

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from stellargraph import datasets

from torch_geometric.utils import from_scipy_sparse_matrix
import torch
from scipy import io
from gensim.models import Word2Vec

data = io.loadmat(fp)
N = from_scipy_sparse_matrix(data['N'])
N = torch.cat((N[0], N[1].view(1, -1)), 0).numpy()
y = data['post_label'].reshape(-1,)
post_nodes = pd.DataFrame(data['post_cate'].todense(), index = data['post_indx'].reshape(-1,))
user_nodes = pd.DataFrame(index = data['user_indx'].reshape(-1,))
edges = pd.DataFrame(
        N.T, columns = ['source', 'target', 'weight']
)
G = StellarGraph(
        {
            'user': user_nodes,
            'post': post_nodes
        }, edges
)
print(G.info())
rw = BiasedRandomWalk(G)

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=1,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=1,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))
str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
model.save("/datasets/dsc180a-wi20-public/Malware/group_data/group_02/sensitive_data/processed/node2vec/node2vec_1_1.model")
print("p = 1, q = 1 saved")

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))
str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
model.save("/datasets/dsc180a-wi20-public/Malware/group_data/group_02/sensitive_data/processed/node2vec/node2vec_05_2.model")
print("p = .5, q = 2 saved")

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=2,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=0.5,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))
str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
model.save("/datasets/dsc180a-wi20-public/Malware/group_data/group_02/sensitive_data/processed/node2vec/node2vec_2_05.model")
print("p = 2, q = 0.5 saved")

