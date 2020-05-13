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
    comm = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'is_submitter', 'parent_id']
                        ) for i in glob(comm)])
    post = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'subreddit']
                        ) for i in glob(post)])
    labl = pd.concat([pd.read_csv(i) for i in glob(labl)])
    labl = labl[labl.label != -1]
    post.author = post.author.str.lower()
    post.subreddit = post.subreddit.str.lower()
    post = post[(post.author != '[deleted]')&(post.author != 'automoderator')& (post.author != 'snapshillbot')]
    post = post[post.id.isin(labl.post_id)]
    comm['parent_id'] = comm.parent_id.str[3:]
    comm = comm[['id','author', 'is_submitter', 'parent_id']]
    comm.columns = ['comment_id','author', 'is_submitter', 'parent_id']
    comm.author = comm.author.str.lower()
    comm = comm[(comm.author != '[deleted]')&(comm.author != 'automoderator') & (comm.author != 'snapshillbot')]
    comm = comm.dropna()
    comm = comm[(comm.parent_id.isin(post.id)) | (comm.parent_id.isin(comm.comment_id))]
    post = post[post.id.isin(comm.parent_id)]
    comm_root = comm[comm.parent_id.str.len() == 6]
    comm_nest = comm[comm.parent_id.str.len() == 7]
    comm_ = comm_root[['comment_id', 'author']]
    print('start preprocessing: (edges)')
    comm_nest_edges = pd.merge(comm_nest, comm_, left_on = 'parent_id', right_on = 'comment_id', how = 'inner')[['author_x', 'author_y']]
    comm_nest_edges.columns = ['who', 'whom']
    comm_root_edges_lower = pd.merge(comm_root, post, left_on = 'parent_id', right_on = 'id', how = 'inner')[['author_x', 'id']]
    comm_root_edges_lower.columns = ['who', 'whom']
    comm_root_edges_upper = pd.merge(comm_root, post, left_on = 'parent_id', right_on = 'id', how = 'inner')[['id', 'author_x']]
    comm_root_edges_upper.columns = ['whom', 'who']
    comm_root_edges = pd.concat([comm_root_edges_lower, comm_root_edges_upper], ignore_index = True)
    edges = pd.concat([comm_nest_edges, comm_root_edges], ignore_index = True)
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
    nodes = pd.merge(nodes, comm, left_on = 'node_name', right_on = 'author', how = 'left')
    nodes = nodes[['node_name', 'subreddit', 'is_submitter','label']]
    nodes.subreddit = nodes.subreddit.fillna('0')
    nodes.label = nodes.label.fillna(-1)
    nodes.is_submitter = nodes.is_submitter.fillna(True).astype(int)
    X = OneHotEncoder(sparse = False).fit_transform(nodes.subreddit.values.reshape(-1, 1))
    X = np.hstack([X, nodes.is_submitter.values.reshape(-1, 1)])
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

