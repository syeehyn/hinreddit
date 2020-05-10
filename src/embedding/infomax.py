import os
import os.path as osp
import sys
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from .utils import create_dataset
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from tqdm import tqdm
import json
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def infomax(fp, PARAMS):
    dataset = create_dataset(fp)
    if PARAMS['CUDA']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    data = dataset[0]
    data.x = data.x.float()
    data = data.to(device)
    model = DeepGraphInfomax(
        hidden_channels=PARAMS['HIDDEN_CHANNELS'], encoder=Encoder(dataset.num_features, PARAMS['SUMMARY']),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    def train():
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()
    losses = []
    for epoch in range(1, PARAMS['NUM_EPOCH']+ 1) :
        loss = train()
        losses.append(loss)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    model.eval()
    with torch.no_grad():
        z, _, _ =model(data.x, data.edge_index)
    if not os.path.exists(os.path.join(fp, 'interim', 'infomax')):
        os.mkdir(os.path.join(fp, 'interim', 'infomax'))
    with open(osp.join(fp, 'interim', 'infomax', 'log.json'), 'w') as f:
        json.dump({'loss': losses}, f)
    torch.save(z, osp.join(fp, 'interim', 'infomax','embedding.pt'))
    torch.save(data, osp.join(fp, 'interim', 'infomax', 'data.pt'))
    return 'embedding infomax created'