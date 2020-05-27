import os.path as osp
import os
import torch
from torch_geometric.nn import MetaPath2Vec
from scipy import io
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import numpy as np
import json

metapath = [
    ('post', 'commented by', 'user'),
    ('user', 'replied by', 'user'),
    ('user', 'wrote', 'post')
]

def metapath2vec(fp, PARAMS):
    g = io.loadmat(osp.join(fp, 'interim', 'graph', PARAMS['GRAPH_NAME']))
    user_user = from_scipy_sparse_matrix(g['U'])
    author_post = from_scipy_sparse_matrix(g['A'])
    post_user = from_scipy_sparse_matrix(g['P'])
    data = Data(
    edge_index_dict = {
        ('user', 'replied by', 'user') : user_user[0],
        ('user', 'wrote', 'post') : author_post[0],
        ('post', 'commented by', 'user') : post_user[0],
    }
    )
    if PARAMS['CUDA']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=PARAMS['EMBEDDING_DIM'],
                    metapath=metapath, walk_length=PARAMS['WALK_LENGTH'], context_size=PARAMS['CONTEXT_SIZE'],
                    walks_per_node=PARAMS['WALKS_PER_NODE'], num_negative_samples=PARAMS['NUM_NEG_SAMPLES'],
                    sparse=True).to(device)
    loader = model.loader(batch_size=PARAMS['BATCH_SIZE'], shuffle=False, num_workers=8)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    def train(epoch, log_steps=1000):
        model.train()
        total_loss = 0
        store = []
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}'))
                store.append(total_loss / log_steps)
                total_loss = 0
        return store
    losses = []
    for epoch in range(1, PARAMS['NUM_EPOCH'] + 1):
        losses.append(train(epoch))
    model.eval()
    with torch.no_grad():
        z = model('post').detach().cpu().numpy()
    if not os.path.exists(os.path.join(fp, 'processed', 'metapath2vec')):
        os.makedirs(os.path.join(fp, 'processed', 'metapath2vec'), exist_ok=True)
    with open(osp.join(fp, 'processed', 'metapath2vec', PARAMS['EMBEDDING_NAME'] + 'log.json'), 'w') as f:
        json.dump({'loss': losses}, f)
    np.save(osp.join(fp, 'processed', 'metapath2vec', PARAMS['EMBEDDING_NAME']), z)
    print('successfully saved embedding')