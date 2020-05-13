import pandas as pd
from glob import glob
import os.path as osp
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import json
from scipy import sparse
import shutil
from .utils import create_dataset
COMM_DIR = osp.join('raw', 'comments', '*.csv')
LABL_DIR = osp.join('interim', 'label', '*.csv')
POST_DIR = osp.join('raw', 'posts', '*.csv')
OUT_DIR = osp.join('interim', 'graph')

def create_graph(fp):
    print('start preprocessing: (filtering)')
    comm = osp.join(fp, COMM_DIR)
    post = osp.join(fp, POST_DIR)
    labl = osp.join(fp, LABL_DIR)
    comm = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'is_submitter', 'link_id']
                        ) for i in glob(comm)])
    post = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'subreddit']
                        ) for i in glob(post)])
    labl = pd.concat([pd.read_csv(i) for i in glob(labl)])
    labl = labl[labl.label != -1]
    post.author = post.author.str.lower()
    post.subreddit = post.subreddit.str.lower()
    post = post[(post.author != '[deleted]')&(post.author != 'automoderator')& (post.author != 'snapshillbot')]
    comm['link_id'] = comm.link_id.str[3:]
    comm = comm[['id','author', 'is_submitter', 'link_id']]
    comm.columns = ['comment_id','author', 'is_submitter', 'link_id']
    comm.author = comm.author.str.lower()
    comm = comm[(comm.author != '[deleted]')&(comm.author != 'automoderator') & (comm.author != 'snapshillbot')]
    comm = comm.dropna()
    comm = comm[(comm.link_id.isin(post.id)) | (comm.link_id.isin(comm.comment_id))]
    post = post[(post.id.isin(labl.post_id)) & (post.id.isin(comm.link_id))]
    author_counts = comm.author.value_counts()
    author_mask = author_counts > 3
    author_counts = author_counts[author_mask].index
    comm = comm[comm.author.isin(author_counts)]
    post = post[post.id.isin(comm.link_id)]
    comm_root = comm.copy()
    print('start preprocessing: (edges)')
    comm_root_edges_lower = pd.merge(comm_root, post, left_on = 'link_id', right_on = 'id', how = 'inner')[['author_x', 'id']]
    comm_root_edges_lower.columns = ['who', 'whom']
    comm_root_edges_upper = pd.merge(comm_root, post, left_on = 'link_id', right_on = 'id', how = 'inner')[['id', 'author_x']]
    comm_root_edges_upper.columns = ['who', 'whom']
    edges = pd.concat([comm_root_edges_lower, comm_root_edges_upper], ignore_index = True)
    graph_idx = np.unique(np.append(edges.who.values, edges.whom.values)).reshape(-1, 1)
    encoder = OrdinalEncoder()
    encoder.fit(graph_idx)
    edges['who_id'] = encoder.transform(edges.who.values.reshape(-1, 1))
    edges['whom_id'] = encoder.transform(edges.whom.values.reshape(-1, 1))
    edge_pairs = edges['who_id'].astype(int).astype(str) + edges['whom_id'].astype(int).astype(str)
    pari_counts = edge_pairs.value_counts().to_frame()
    edge_pairs = edge_pairs.to_frame()
    edge_weights = pd.merge(edge_pairs, pari_counts, left_on = 0, right_index = True, how = 'left')['0_y'].values
    edges['weights'] = (edge_weights - edge_weights.mean()) / edge_weights.std()
    edges_ = edges[['who_id', 'whom_id', 'weights']].drop_duplicates().sort_values(['who_id', 'whom_id'])
    edge_idx = edges_[['who_id', 'whom_id']].values.astype(int)
    edge_weight = edges_[['weights']].values
    post_label = pd.merge(post, labl, left_on = 'id', right_on = 'post_id', how = 'left')
    print('start preprocessing: (nodes)')
    nodes = pd.DataFrame(
        {
            'node_name': pd.Series(encoder.categories_[0])
        }
    )
    nodes = pd.merge(nodes, post_label, left_on = 'node_name', right_on = 'id', how = 'left')
    nodes = nodes[['node_name', 'subreddit','label']]
    nodes.subreddit = nodes.subreddit.fillna('0')
    nodes.label = nodes.label.fillna(-1)
    X = OneHotEncoder(sparse = False).fit_transform(nodes.subreddit.values.reshape(-1, 1))
    y = nodes.label.values.reshape(-1, 1)
    post_mask = (nodes.label.values != -1).reshape(-1, 1)
    node_output = np.hstack([X, post_mask, y])
    edge_output = np.hstack([edge_idx, edge_weight])
    print('writing raw graphs')
    np.save(osp.join(fp, OUT_DIR, 'nodes.npy'), node_output)
    np.save(osp.join(fp, OUT_DIR, 'edges.npy'), edge_output)
    print('formatting to pytorch dataset')
    shutil.rmtree(osp.join(fp, 'interim', 'graph', 'processed'), ignore_errors = True)
    shutil.rmtree(osp.join(fp, 'interim', 'graph', 'raw'), ignore_errors = True)
    create_dataset(fp)
    print('graph with {} nodes constructed'.format(X.shape[0]))

