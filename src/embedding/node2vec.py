import os
import os.path as osp
import sys
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from scipy import io
from tqdm import tqdm
import json
import shutil
def node2vec(fp, PARAMS):
    N = io.loadmat(osp.join(fp, 'interim', 'graph', PARAMS['GRAPH_NAME']))['N']
    edge_idx, x =from_scipy_sparse_matrix(N)
    post_indx = io.loadmat(osp.join(fp, 'interim', 'graph', PARAMS['GRAPH_NAME']))['post_indx']
    post_indx = torch.from_numpy(post_indx.reshape(-1,))
    data = Data(
        x = x,
        edge_index = edge_idx
    )
    print(data.num_nodes)
    print(data)
    loader = DataLoader(post_indx, batch_size=PARAMS['BATCH_SIZE'], shuffle=False)
    if PARAMS['CUDA']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=PARAMS['EMBEDDING_DIM'], walk_length=PARAMS['WALK_LENGTH'],
                    context_size=PARAMS['CONTEXT_SIZE'], walks_per_node=PARAMS['WALKS_PER_NODE'], p = PARAMS['P'], q = PARAMS['Q'], sparse=True)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
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
    print('number of nodes to be embedded {}'.format(len(post_indx)))
    print('Start Node2vec Embedding Process with Following Parameters:')
    print(PARAMS)
    losses = []
    for epoch in range(1, PARAMS['NUM_EPOCH'] + 1):
        loss = train()
        losses.append(loss)
        print('Epoch: {:02d}, Node2vec Loss: {:.4f}'.format(epoch, loss))
    model.eval()
    with torch.no_grad():
        z = model(post_indx.to(device))
    if not os.path.exists(os.path.join(fp, 'processed', 'node2vec')):
        os.makedirs(os.path.join(fp, 'processed', 'node2vec'), exist_ok=True)
    with open(osp.join(fp, 'processed', 'node2vec', 'log_1.json'), 'w') as f:
        json.dump({'loss': losses}, f)
    z = z.detach().cpu().numpy()
    np.save(osp.join(fp, 'processed', 'node2vec', PARAMS['EMBEDDING_NAME']), z)
    print('successfully saved embedding')

