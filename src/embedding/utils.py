import os
import pandas as pd
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
        return ['nodes.csv', 'edges.csv']

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
        nodes = pd.read_csv(os.path.join(self.root, 'nodes.csv'))
        edges = pd.read_csv(os.path.join(self.root, 'edges.csv'))
        
        y = torch.from_numpy(nodes['label'].values.astype(int))
        edge_index = torch.from_numpy(edges.values.T).long()
        post_mask = torch.from_numpy(nodes['is_post'].astype(bool).values)
        data_list.append(dt(
                        edge_index = edge_index,  
                        y = y, 
                        post_mask = post_mask))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_dataset(fp):
    data = RedditData(root = os.path.join(fp, 'interim', 'graph'))
    return data
