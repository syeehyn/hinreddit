import pandas as pd
from glob import glob
import os.path as osp
import os
import numpy as np
import json
import shutil
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from stellargraph import datasets
from IPython.display import display, HTML
from gensim.models import Word2Vec
from tensorflow import keras
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

COMM_DIR = osp.join('raw', 'comments', '*.csv')
LABL_DIR = osp.join('interim', 'label', '*.csv')
POST_DIR = osp.join('raw', 'posts', '*.csv')
OUT_DIR = osp.join('interim', 'graph')

def create_graph(fp, subreddit):
    comm = osp.join(fp, COMM_DIR)
    post = osp.join(fp, POST_DIR)
    labl = osp.join(fp, LABL_DIR)
    comm = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'parent_id', 'link_id']
                        ) for i in glob(comm)])
    post = pd.concat([pd.read_csv(i, usecols = ['id', 'author', 'subreddit']
                        ) for i in glob(post)])
    labl = pd.concat([pd.read_csv(i) for i in glob(labl)])
    labl = labl[labl.label != -1]
    post.author = post.author.str.lower()
    post.subreddit = post.subreddit.str.lower()
    post = post[(post.author != '[deleted]')&(post.author != 'automoderator')& (post.author != 'snapshillbot')]
    comm['parent_id'] = comm.parent_id.str[3:]
    comm['link_id'] = comm.link_id.str[3:]
    comm = comm[['id','author', 'parent_id', 'link_id']]
    comm.author = comm.author.str.lower()
    comm = comm[(comm.author != '[deleted]')&(comm.author != 'automoderator') & (comm.author != 'snapshillbot')]
    comm = comm.dropna()
    post = post[post.subreddit == subreddit]
    post = post[(post.id.isin(labl.post_id)) & (post.id.isin(comm.link_id))]
    comm = comm[comm.link_id.isin(post.id)]
    comm = comm[(comm.parent_id.isin(post.id)) | (comm.parent_id.isin(comm.id)) | (comm.link_id.isin(post.id))]
    comm_root = comm[comm.parent_id.str.len() == 6]
    comm_nest = comm[comm.parent_id.str.len() == 7]
    print('start preprocessing: (edges)')
    post_pauthor = post[['id', 'author']]
    post_pauthor.columns = ['who', 'whom']
    pauthor_post = post[['id', 'author']]
    pauthor_post.columns = ['whom', 'who']
    post_author_edges = pd.concat([post_pauthor, pauthor_post], ignore_index = True)
    post_comment_edges_root = comm_root[['parent_id', 'author']]
    post_comment_edges_root.columns = ['who', 'whom']
    post_comment_edges_nest = comm_nest[['link_id', 'author']]
    post_comment_edges_nest.columns = ['who', 'whom']
    user_user_edges = pd.merge(comm_nest[['author', 'parent_id']], comm_nest[['author', 'id']], \
    how='left', left_on='parent_id', right_on='id').dropna()[['author_x', 'author_y']]
    user_user_edges.columns = ['whom', 'who']
    edges = pd.concat([post_author_edges, post_comment_edges_root, post_comment_edges_nest, user_user_edges], \
                        ignore_index=True)
    edges = edges[~((edges.who.isin(post.id))&(edges.whom.isin(post.id)))]
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
    pair_counts = pair_counts.astype(int).to_frame()
    edges = pd.merge(edge_pair, pair_counts, left_on = 'pairs', right_index = True)[['who', 'whom', 0]]
    edge_idx = edges.drop_duplicates()
    edge_idx = pd.merge(edge_idx, node_maps, left_on='who', right_on='name', how='left')[['id', 'whom', 0]]
    edge_idx.columns = ['who_id', 'whom', 'weight']
    edge_idx = pd.merge(edge_idx, node_maps, left_on='whom', right_on='name', how='left')[['who_id', 'id', 'weight']]
    edge_idx.columns = ['who_id', 'whom_id', 'weight']
    edge_weight = edge_idx[['weight']].values
    edge_idx = edge_idx.sort_values(['who_id', 'whom_id'])
    edge_idx_ = edge_idx[['who_id', 'whom_id']].values
    post_mask = node_maps.name.isin(post_names)
    post_indx = node_maps[post_mask].id.values
    y = pd.merge(node_maps[post_mask][['name']], 
                labl, left_on='name', right_on='post_id', how='left').label.values
    print('done')
    return edge_idx_, edge_weight.reshape(-1), y, subreddit, post_indx

def plot_embedding(embd, y):
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(embd)
    fig = plt.figure(figsize=(16, 16))
    for i in range(2):
        plt.scatter(node_embeddings_2d[y == i, 0], node_embeddings_2d[y == i, 1], s=10, label = i)
    plt.legend()
    plt.axis('off')
    plt.show()
    return fig
def evaluate(clf, X_train, y_train, X_test, y_test):
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]   
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    res = {}
    for i in METRICS:
        res[i.name] = i(y_test, y_pred).numpy()
    return res

def generate_node2vec_result(edge_idx, edge_weight, y, subreddit, post_indx, walk_length, p, q):
    edges = pd.DataFrame(edge_idx, columns = ['source', 'target'])
    edges['weight'] = edge_weight.reshape(-1)
    G = StellarGraph(edges=edges, is_directed=True)
    print(G.info())
    rw = BiasedRandomWalk(G)
    print('Random Walk Length: {}, p: {}, q: {}'.format(walk_length, p, q))
    walks = rw.run(
        nodes=list(G.nodes()),  # root nodes
        length=100,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=1,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=.5,  # Defines (unormalised) probability, 1/q, for moving away from source node
    )
    print("Number of random walks: {}".format(len(walks)))
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, size=128, window=5, min_count=0, sg=1, workers=8, iter=1)
    embd = model.wv.vectors[post_indx, :]
    fig = plot_embedding(embd, y)
    neg, pos = np.bincount(y)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    print('Test Size: {}'.format(0.2))
    X_train, X_test, y_train, y_test = train_test_split(embd, y, test_size=0.2)
    clfs = [
        LogisticRegression(
        verbose=False, max_iter=1000, class_weight = class_weight
        ),
        LinearSVC(class_weight=class_weight),
        RandomForestClassifier(class_weight=class_weight)
    ]
    res = {}
    for clf in clfs:
        res[clf.__class__.__name__] = evaluate(clf, X_train, y_train, X_test, y_test)
    return fig, res