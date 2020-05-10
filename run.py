import json
import sys
from src.etl import *
from src.embedding import create_graph, node2vec
from src.utils import evaluate
from src.models import *
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from p_tqdm import p_map
TESTPARAMS = json.load(open('config/test-params.json'))
EDAPARAMS = json.load(open('config/eda-params.json'))
DATAPARAMS = json.load(open('config/data-params.json'))
TESTDIR = TESTPARAMS['META_ARGS']['filepath']
EDADIR = EDAPARAMS['META_ARGS']['filepath']
DATADIR = DATAPARAMS['META_ARGS']['filepath']
MODELDIR = 'config/nlp_model.zip'
DATA_NODE2VEC = json.load(open('config/embedding/node2vec.json'))
TEST_NODE2VEC = json.load(open('config/embedding/test-node2vec.json'))

def env_test():
    if not os.path.exists(TESTDIR):
        os.mkdir(TESTDIR)
    if not os.path.exists(os.path.join(TESTDIR, 'raw')):
        os.mkdir(os.path.join(TESTDIR, 'raw'))
    if not os.path.exists(os.path.join(TESTDIR, 'raw', 'posts')):
        os.mkdir(os.path.join(TESTDIR, 'raw', 'posts'))
    if not os.path.exists(os.path.join(TESTDIR, 'raw', 'posts_detail')):
        os.mkdir(os.path.join(TESTDIR, 'raw', 'posts_detail'))
    if not os.path.exists(os.path.join(TESTDIR, 'raw', 'comments')):
        os.mkdir(os.path.join(TESTDIR, 'raw', 'comments'))
    if not os.path.exists(os.path.join(TESTDIR, 'interim')):
        os.mkdir(os.path.join(TESTDIR, 'interim'))
    if not os.path.exists(os.path.join(TESTDIR, 'interim', 'label')):
        os.mkdir(os.path.join(TESTDIR, 'interim', 'label'))
    if not os.path.exists(os.path.join(TESTDIR, 'interim', 'label', 'post')):
        os.mkdir(os.path.join(TESTDIR, 'interim', 'label', 'post'))
    if not os.path.exists(os.path.join(TESTDIR, 'interim', 'label', 'comment')):
        os.mkdir(os.path.join(TESTDIR, 'interim', 'label', 'comment'))
    if not os.path.exists(os.path.join(TESTDIR, 'interim', 'graph')):
        os.mkdir(os.path.join(TESTDIR, 'interim', 'graph'))
    return
def env_data():
    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)
    if not os.path.exists(os.path.join(DATADIR, 'raw')):
        os.mkdir(os.path.join(DATADIR, 'raw'))
    if not os.path.exists(os.path.join(DATADIR, 'raw', 'posts')):
        os.mkdir(os.path.join(DATADIR, 'raw', 'posts'))
    if not os.path.exists(os.path.join(DATADIR, 'raw', 'posts_detail')):
        os.mkdir(os.path.join(DATADIR, 'raw', 'posts_detail'))
    if not os.path.exists(os.path.join(DATADIR, 'raw', 'comments')):
        os.mkdir(os.path.join(DATADIR, 'raw', 'comments'))
    if not os.path.exists(os.path.join(DATADIR, 'interim')):
        os.mkdir(os.path.join(DATADIR, 'interim'))
    if not os.path.exists(os.path.join(DATADIR, 'interim', 'label')):
        os.mkdir(os.path.join(DATADIR, 'interim', 'label'))
    if not os.path.exists(os.path.join(DATADIR, 'interim', 'label', 'post')):
        os.mkdir(os.path.join(DATADIR, 'interim', 'label', 'post'))
    if not os.path.exists(os.path.join(DATADIR, 'interim', 'label', 'comment')):
        os.mkdir(os.path.join(DATADIR, 'interim', 'label', 'comment'))
    if not os.path.exists(os.path.join(DATADIR, 'interim', 'graph')):
        os.mkdir(os.path.join(DATADIR, 'interim', 'graph'))
    return

def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    else:
        env_data()
    # if 'test' in targets:
    #     fetch_submissions(**TESTPARAMS)
    #     submissions_detail(TESTDIR)
    #     comments_detail(TESTDIR)
    if 'data' in targets:
        fetch_submissions(**DATAPARAMS)
        submissions_detail(DATADIR)
        comments_detail(DATADIR)
    if 'sentimental' in targets:
        model, tokenizer = load_nlp('config/nlp_model.zip', DATADIR)
        label_comments(DATADIR, model, tokenizer)
        label_posts(DATADIR, model, tokenizer)
    if 'label' in targets:
        labeling(DATADIR)
    if 'graph' in targets:
        create_graph(DATADIR)
    if 'embedding' in targets:
        node2vec(DATADIR, DATA_NODE2VEC)
#=================For test============================#
    if 'data-test' in targets:
        fetch_submissions(**TESTPARAMS)
        submissions_detail(TESTDIR)
        comments_detail(TESTDIR)
    if 'sentimental-test' in targets:
        model, tokenizer = load_nlp('config/nlp_model.zip', TESTDIR)
        label_comments(TESTDIR, model, tokenizer)
        label_posts(TESTDIR, model, tokenizer)
    if 'label-test' in targets:
        labeling(TESTDIR)
    if 'graph-test' in targets:
        create_graph(TESTDIR)
    if 'embedding-test' in targets:
        node2vec(TESTDIR, TEST_NODE2VEC)



if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)