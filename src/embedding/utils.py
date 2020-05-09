import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data as dt
def Data(fp):
    nodes = pd.read_csv(os.path.join(fp, 'interim', 'graph_table', 'nodes.csv'))
    edges = pd.read_csv(os.path.join(fp, 'interim', 'graph_table', 'edges.csv'))
    onehot = OneHotEncoder()
    subreddit_feature = onehot.fit_transform(nodes[['subreddit']].values)
    x = torch.from_numpy(np.hstack([nodes[['is_submitter']].values, subreddit_feature.todense()])).long()
    y = torch.from_numpy(nodes['label'].values.astype(int))
    edge_index = torch.from_numpy(edges.values.T).long()
    post_mask = torch.from_numpy(nodes['is_post'].astype(bool).values)
    np.random.seed(0)
    data = dt(x = x, 
                edge_index = edge_index,  
                y = y, 
                post_mask = post_mask)
    return data