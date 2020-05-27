import json
import sys
from src.etl import *
from src.embedding import *
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
G1_NODE2VEC = json.load(open('config/embedding/graph_1/node2vec.json'))
G2_NODE2VEC = json.load(open('config/embedding/graph_2/node2vec.json'))
G1_TEST_NODE2VEC = json.load(open('config/embedding/graph_1/test-node2vec.json'))
G2_TEST_NODE2VEC = json.load(open('config/embedding/graph_2/test-node2vec.json'))
G1_INFOMAX = json.load(open('config/embedding/graph_1/infomax.json'))
G2_INFOMAX = json.load(open('config/embedding/graph_2/infomax.json'))
G1_TEST_INFOMAX = json.load(open('config/embedding/graph_1/test-infomax.json'))
G2_TEST_INFOMAX = json.load(open('config/embedding/graph_2/test-infomax.json'))

def env(fp):
    os.makedirs(os.path.join(fp, 'raw', 'posts'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'raw', 'posts_detail'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'raw', 'comments'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'interim', 'label', 'post'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'interim', 'label', 'comment'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'interim', 'graph'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'interim', 'graph'), exist_ok=True)
    os.makedirs(os.path.join(fp, 'processed'), exist_ok=True)
    return

def main(targets):
    if any(['test'in i for i in targets]):
        env(TESTDIR)
    else:
        env(DATADIR)
    if 'data' in targets:
        fetch_submissions(**DATAPARAMS)
        submissions_detail(DATADIR)
        comments_detail(DATADIR)
    if 'label' in targets:
        model, tokenizer = load_nlp('config/nlp_model.zip', DATADIR)
        label_comments(DATADIR, model, tokenizer)
        label_posts(DATADIR, model, tokenizer)
        labeling(DATADIR)
    if 'baseline' in targets:
        posts = extract_feat(os.path.join(DATADIR, 'raw', 'posts'),\
                            os.path.join(DATADIR, 'interim', 'label', 'label.csv'))
        baseline_model(posts)
    if 'graph' in targets:
        g1(DATADIR)
        g2(DATADIR)
    if 'node2vec' in targets:
        node2vec(DATADIR, G1_NODE2VEC)
        node2vec(DATADIR, G2_NODE2VEC)
    if 'infomax' in targets:
        infomax(DATADIR, G1_INFOMAX)
        infomax(DATADIR, G2_INFOMAX)
    if 'debug' in targets:
        infomax(DATADIR, G1_INFOMAX)
        # node2vec(DATADIR, G2_NODE2VEC)
#=================For test============================#
    if 'data-test' in targets:
        fetch_submissions(**TESTPARAMS)
        submissions_detail(TESTDIR)
        comments_detail(TESTDIR)
    if 'label-test' in targets:
        model, tokenizer = load_nlp('config/nlp_model.zip', TESTDIR)
        label_comments(TESTDIR, model, tokenizer)
        label_posts(TESTDIR, model, tokenizer)
        labeling(TESTDIR)
    if 'baseline-test' in targets:
        posts = extract_feat(os.path.join(TESTDIR, 'raw', 'posts'),\
                            os.path.join(TESTDIR, 'interim', 'label', 'label.csv'))
        baseline_model(posts)
    if 'graph-test' in targets:
        g1(TESTDIR)
        g2(TESTDIR)
    if 'node2vec-test' in targets:
        node2vec(TESTDIR, G1_TEST_NODE2VEC)
        node2vec(TESTDIR, G2_TEST_NODE2VEC)
    if 'infomax-test' in targets:
        infomax(TESTDIR, G1_TEST_INFOMAX)
        infomax(TESTDIR, G2_TEST_INFOMAX)
    if 'test-project' in targets:
        ##
        fetch_submissions(**TESTPARAMS)
        submissions_detail(TESTDIR)
        comments_detail(TESTDIR)
        ##
        model, tokenizer = load_nlp('config/nlp_model.zip', TESTDIR)
        label_comments(TESTDIR, model, tokenizer)
        label_posts(TESTDIR, model, tokenizer)
        labeling(TESTDIR)
        ##
        g1(TESTDIR)
        g2(TESTDIR)
        ##
        node2vec(TESTDIR, G1_TEST_NODE2VEC)
        node2vec(TESTDIR, G2_TEST_NODE2VEC)
        infomax(TESTDIR, G1_TEST_INFOMAX)
        infomax(TESTDIR, G2_TEST_INFOMAX)




if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)