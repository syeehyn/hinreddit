import pandas as pd
from glob import glob
import os.path as osp
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import json
from scipy import sparse
import shutil
from .utils import create_dataset
import scipy.io as io

COMM_DIR = osp.join('raw', 'comments', '*.csv')
LABL_DIR = osp.join('interim', 'label', '*.csv')
POST_DIR = osp.join('raw', 'posts', '*.csv')
OUT_DIR = osp.join('interim', 'graph')

def create_graph(fp):
    print('start preprocessing: (filtering)')
    try:
        os.remove(osp.join(fp, OUT_DIR, 'graph.mat'))
    except FileNotFoundError:
        pass
    comm = osp.join(fp, COMM_DIR)
    post = osp.join(fp, POST_DIR)
    labl = osp.join(fp, LABL_DIR)
    comm = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'parent_id']
                        ) for i in glob(comm)])
    post = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'subreddit']
                        ) for i in glob(post)])
    labl = pd.concat([pd.read_csv(i) for i in glob(labl)])
    labl = labl[labl.label != -1]
    post.author = post.author.str.lower()
    post.subreddit = post.subreddit.str.lower()
    # post = post[(post.author != '[deleted]')&(post.author != 'automoderator')& (post.author != 'snapshillbot')]
    comm['parent_id'] = comm.parent_id.str[3:]
    comm = comm[['id','author', 'parent_id']]
    comm.author = comm.author.str.lower()
    # comm = comm[(comm.author != '[deleted]')&(comm.author != 'automoderator') & (comm.author != 'snapshillbot')]
    comm = comm.dropna()
    post = post[(post.id.isin(labl.post_id)) & (post.id.isin(comm.parent_id))]
    comm = comm[(comm.parent_id.isin(post.id)) | (comm.parent_id.isin(comm.id))]
    # author_counts = comm.author.value_counts()
    # author_mask = author_counts > 3
    # author_counts = author_counts[author_mask].index
    comm_root = comm[comm.parent_id.str.len() == 6]
    comm_nest = comm[comm.parent_id.str.len() == 7]
    print('start preprocessing: (edges)')
    post_pauthor = post[['id', 'author']]
    post_pauthor.columns = ['who', 'whom']
    pauthor_post = post[['id', 'author']]
    pauthor_post.columns = ['whom', 'who']
    pauthor_post_edges = pd.concat([post_pauthor, pauthor_post], ignore_index = True)
    cauthor_pauthor_edges = pd.merge(comm_root[['author', 'parent_id']], post[['id', 'author']], \
            left_on = 'parent_id', right_on = 'id', how = 'left')[['author_x', 'author_y']]
    cauthor_pauthor_edges.columns = ['who', 'whom']
    cauthor_cauthor = pd.merge(comm_nest[['parent_id', 'author']], comm_nest[['id', 'author']], \
            left_on = 'parent_id', right_on = 'id', how = 'left')[['author_x', 'author_y']]
    cauthor_cauthor.columns = ['who', 'whom']
    edges = pd.concat([pauthor_post_edges, cauthor_pauthor_edges, cauthor_cauthor], ignore_index = True)
    edges = edges[edges.who != edges.whom]
    print('start preprocessing: (nodes)')
    post_names = post.id.unique()
    node_names = np.unique(np.append(edges.who.dropna().values, edges.whom.dropna().values))
    node_maps = pd.DataFrame(
                {
                    'id': np.arange(len(node_names)),
                    'name': node_names
                }
    )
    edge_pairs = edges.dropna().who + edges.dropna().whom
    edge_pair = edges.copy()
    pair_counts = edge_pairs.value_counts()
    edge_pair['pairs'] = edge_pairs
    pair_counts = np.log(pair_counts + pair_counts.std()).to_frame()
    edges = pd.merge(edge_pair, pair_counts, left_on = 'pairs', right_index = True)[['who', 'whom', 0]]
    edge_idx = edges.drop_duplicates()
    edge_idx = pd.merge(edge_idx, node_maps, left_on='who', right_on='name', how='left')[['id', 'whom', 0]]
    edge_idx.columns = ['who_id', 'whom', 'weight']
    edge_idx = pd.merge(edge_idx, node_maps, left_on='whom', right_on='name', how='left')[['who_id', 'id', 'weight']]
    edge_idx.columns = ['who_id', 'whom_id', 'weight']
    edge_weight = edge_idx[['weight']].values
    edge_idx = edge_idx.sort_values(['who_id', 'whom_id'])
    edge_idx_ = edge_idx[['who_id', 'whom_id']].values
    N = sparse.csr_matrix((edge_weight.reshape(-1,), (edge_idx_[:, 0], edge_idx_[:, 1])), \
                                shape = (node_maps.shape[0], node_maps.shape[0]))
    post_mask = node_maps.name.isin(post_names)
    post_indx = node_maps[post_mask].id.values
    user_indx = node_maps[~post_mask].id.values
    U = N[user_indx, :][:, user_indx]
    PU = N[post_indx, :][:, user_indx]
    UP = N[user_indx, :][:, post_indx]
    N_edge = np.vstack(N.nonzero())
    N_edge_weight = edge_weight
    U_edge = np.vstack(U.nonzero())
    U_edge_weight = edge_idx[edge_idx.who_id.isin(user_indx)&edge_idx.whom_id.isin(user_indx)][['weight']].values
    PU_edge = np.vstack(PU.nonzero())
    PU_edge_weight = edge_idx[edge_idx.who_id.isin(post_indx)&edge_idx.whom_id.isin(user_indx)][['weight']].values
    UP_edge = np.vstack(UP.nonzero())
    UP_edge_weight = edge_idx[edge_idx.who_id.isin(user_indx)&edge_idx.whom_id.isin(post_indx)][['weight']].values
    subreddit = pd.merge(node_maps[post_mask][['name']], post[['id', 'subreddit']].drop_duplicates(), \
                        left_on = 'name', right_on='id', how='left').subreddit.values
    heter_feature = OneHotEncoder().fit_transform(subreddit.reshape(-1, 1))
    heter_label = pd.merge(node_maps[post_mask][['name']], 
            labl, left_on='name', right_on='post_id', how='left').label.values
    print('start writing matrices')
    res = {}
    res['N'] = N
    res['N_edge'] = N_edge
    res['N_edge_weight'] = N_edge_weight
    res['U'] = U
    res['U_edge'] = U_edge
    res['U_edge_weight'] = U_edge_weight
    res['PU'] = PU
    res['PU_edge'] = PU_edge
    res['PU_edge_weight'] = PU_edge_weight
    res['UP'] = UP
    res['UP_edge'] = UP_edge
    res['UP_edge_weight'] = UP_edge_weight
    res['P_label'] = heter_label
    res['P_cate'] = heter_feature
    res['P_indx'] = post_indx
    res['U_indx'] = user_indx
    io.savemat(osp.join(fp, OUT_DIR, 'graph.mat'), res)
    print('graph constructed, with N shape: {}, N edges: {}, PU shape: {}'.format(N.shape, N_edge.shape[1], PU.shape),)

