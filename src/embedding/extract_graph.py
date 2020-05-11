import pandas as pd
from glob import glob
import os.path as osp
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import json
from scipy import sparse

COMM_DIR = osp.join('raw', 'comments', '*.csv')
LABL_DIR = osp.join('interim', 'label', '*.csv')
POST_DIR = osp.join('raw', 'posts', '*.csv')
OUT_DIR = osp.join('interim', 'graph')

def create_graph(fp):
    comm = osp.join(fp, COMM_DIR)
    post = osp.join(fp, POST_DIR)
    labl = osp.join(fp, LABL_DIR)
    comm = pd.concat([pd.read_csv(i, usecols = ['author', 'link_id', 'subreddit']
                        ) for i in glob(comm)])
    post = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'subreddit']
                        ) for i in glob(post)])
    labl = pd.concat([pd.read_csv(i) for i in glob(labl)])
    labl = labl[labl.label != -1]
    post.author = post.author.str.lower()
    post.subreddit = post.subreddit.str.lower()
    post = post[(post.author != '[deleted]')&(post.author != 'automoderator')& (post.author != 'snapshillbot')]
    post = post[post.id.isin(labl.post_id)]
    comm['post_id'] = comm.link_id.str[3:]
    comm = comm[['author', 'subreddit', 'post_id']]
    comm.author = comm.author.str.lower()
    comm.subreddit = comm.subreddit.str.lower()
    comm = comm[(comm.author != '[deleted]')&(comm.author != 'automoderator') & (comm.author != 'snapshillbot')]
    author_counts = comm.author.value_counts()
    author_mask = author_counts > 3
    author_counts = author_counts[author_mask].index
    comm = comm[comm.author.isin(author_counts)]
    comm = comm[comm.post_id.isin(post.id)]
    add_comm = post.copy()
    add_comm.columns = ['post_id', 'author', 'subreddit']
    add_comm =add_comm[['author', 'subreddit', 'post_id']].drop_duplicates()
    comm = pd.concat([comm, add_comm], ignore_index = True).drop_duplicates()
    post.columns = ['node_id', 'author', 'subreddit']
    post = post.drop_duplicates()
    comm.columns = ['node_id', 'subreddit', 'parent_id']
    comm['is_post'] = False
    post['is_post'] = True
    graph = pd.concat([
                post[['node_id', 'is_post']],
                comm[['node_id', 'is_post']]
                ]).drop_duplicates().reset_index(drop = True)
    graph = pd.merge(graph, labl, left_on = 'node_id', right_on = 'post_id', how = 'left')
    graph = graph[['node_id', 'is_post', 'label']]
    graph = graph.fillna(-1)
    encoder = OrdinalEncoder()
    graph.node_id = encoder.fit_transform(graph[['node_id']])
    post.node_id = encoder.transform(post[['node_id']])
    comm.node_id = encoder.transform(comm[['node_id']])
    comm.parent_id = encoder.transform(comm[['parent_id']])
    graph = graph.sort_values('node_id')
    edge_index = comm[['node_id', 'parent_id']].drop_duplicates().sort_values('node_id')
    graph.to_csv(osp.join(fp, OUT_DIR, 'nodes.csv'), header = True, index = False)
    edge_index.to_csv(osp.join(fp, OUT_DIR, 'edges.csv'), header = True, index = False)
    with open(osp.join(fp, OUT_DIR, 'nodes_info.json'), 'w') as f:
        json.dump(pd.Series(encoder.categories_[0]).to_dict(), f)
    post = graph[graph.is_post.astype(bool)]
    user = graph[~graph.is_post.astype(bool)]
    post_id = post.node_id.values
    user_id = user.node_id.values
    post_map = pd.DataFrame(
            {
                'parent_id': post_id,
                'post': np.arange(len(post_id))
            }
    )
    user_map = pd.DataFrame(
            {
                'node_id': user_id,
                'user': np.arange(len(user_id))
            }
    )
    mat_indx = edge_index.copy()
    mat_indx = pd.merge(mat_indx, user_map, on = 'node_id', how = 'left')
    mat_indx = pd.merge(mat_indx, post_map, on = 'parent_id', how = 'left')
    mat_indx = mat_indx[['post', 'user']].sort_values(['post', 'user'])
    mat_indx = mat_indx.values
    adj_matrix = sparse.csc_matrix((np.ones(mat_indx.shape[0]), (mat_indx[:, 0], mat_indx[:, 1])))
    sparse.save_npz(osp.join(fp, OUT_DIR, 'adj_matrix.npz'), adj_matrix)