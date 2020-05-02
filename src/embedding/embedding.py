import os.path as osp
import sys
import torch
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from .embedding import Data
from torch_geometric.nn import Node2Vec

def embedding(fp):
    data = Data(fp)
    loader = DataLoader(torch.arange(data.num_nodes), batch_size=128, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.num_nodes, embedding_dim=256, walk_length=20,
                    context_size=10, walks_per_node=10)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    def train():
        model.train()
        total_loss = 0
        for subset in loader:
            optimizer.zero_grad()
            loss = model.loss(data.edge_index, subset.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    print('batch size 128, embedding dim 245, walk_length 20, walks per node 10')
    for epoch in range(1, 5):
        loss = train()
        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
    model.eval()
    with torch.no_grad():
        z = model(torch.arange(data.num_nodes, device=device))
    torch.save(z, osp.join(fp, 'interim', 'embedding','embedding.pt'))
    torch.save(z, osp.join(fp, 'interim', 'embedding', 'data.pt'))
    return 'embedding created'