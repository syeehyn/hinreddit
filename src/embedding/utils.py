import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data as dt
from torch_geometric.data import InMemoryDataset
class RedditData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RedditData, self).__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nodes.npy', 'edges.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        nodes = np.load(os.path.join(self.root, 'nodes.npy'))
        edges = np.load(os.path.join(self.root, 'edges.npy'))
        X, post_mask, y= nodes[:, :-2], nodes[:, -2], nodes[:, -1]
        edge_index, edge_weights = edges[:, :-1], edges[:, -1]
        data_list.append(
            dt(
                x = torch.from_numpy(X).long(),
                edge_index = torch.from_numpy(edge_index.T).long(),
                y = torch.from_numpy(y).int(),
                post_mask = torch.from_numpy(post_mask.astype(bool)),
                edge_attr = torch.from_numpy(edge_weights).view(-1, 1)
            )
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_dataset(fp):
    data = RedditData(root = os.path.join(fp, 'interim', 'graph'))
    return data
