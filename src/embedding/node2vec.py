import os
import os.path as osp
import sys
import torch
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from .utils import create_dataset
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import json

def node2vec(fp, PARAMS):
    dataset = create_dataset(fp)
    data = dataset[0]
    loader = DataLoader(torch.arange(data.num_nodes), batch_size=PARAMS['BATCH_SIZE'], shuffle=False)
    if PARAMS['CUDA']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    model = Node2Vec(data.num_nodes, embedding_dim=PARAMS['EMBEDDING_DIM'], walk_length=PARAMS['WALK_LENGTH'],
                    context_size=PARAMS['CONTEXT_SIZE'], walks_per_node=PARAMS['WALKS_PER_NODE'])
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    def train():
        model.train()
        total_loss = 0
        for subset in tqdm(loader):
            optimizer.zero_grad()
            loss = model.loss(data.edge_index, subset.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    print('Start Node2vec Embedding Process with Following Parameters:')
    print(PARAMS)
    losses = []
    for epoch in range(1, PARAMS['NUM_EPOCH'] + 1):
        loss = train()
        losses.append(loss)
        print('Epoch: {:02d}, Node2vec Loss: {:.4f}'.format(epoch, loss))
    model.eval()
    with torch.no_grad():
        z = model(torch.arange(data.num_nodes, device=device))
    if not os.path.exists(os.path.join(fp, 'interim', 'node2vec')):
        os.mkdir(os.path.join(fp, 'interim', 'node2vec'))
    with open(osp.join(fp, 'interim', 'node2vec', 'log.json'), 'w') as f:
        json.dump({'loss': losses}, f)
    torch.save(z, osp.join(fp, 'interim', 'node2vec','embedding.pt'))
    torch.save(data, osp.join(fp, 'interim', 'node2vec', 'data.pt'))
    return 'embedding created'