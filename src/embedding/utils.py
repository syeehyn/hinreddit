import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data
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
    mask_ind = np.random.choice(range(nodes.shape[0]), int(nodes.shape[0] * .2), replace=False)
    train_mask, test_mask = np.ones(nodes.shape[0], dtype = bool), np.zeros(nodes.shape[0], dtype = bool)
    train_mask[mask_ind] = 0
    test_mask[test_mask] = 1
    train_mask, test_mask = torch.from_numpy(train_mask), torch.from_numpy(test_mask)
    data = Data(x = x, 
            edge_index = edge_index,  
            y = y, 
            train_mask = train_mask, 
            test_mask = test_mask,
            post_mask = post_mask)
    return data