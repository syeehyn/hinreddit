import json
import sys
from src.etl import fetch_submissions, submissions_detail, comments_detail
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
    return
def env_eda():
    if not os.path.exists(EDADIR):
        os.mkdir(EDADIR)
    if not os.path.exists(os.path.join(EDADIR, 'raw')):
        os.mkdir(os.path.join(EDADIR, 'raw'))
    if not os.path.exists(os.path.join(EDADIR, 'raw', 'posts')):
        os.mkdir(os.path.join(EDADIR, 'raw', 'posts'))
    if not os.path.exists(os.path.join(EDADIR, 'raw', 'posts_detail')):
        os.mkdir(os.path.join(EDADIR, 'raw', 'posts_detail'))
    if not os.path.exists(os.path.join(EDADIR, 'raw', 'comments')):
        os.mkdir(os.path.join(EDADIR, 'raw', 'comments'))
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
    return

def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    if any(['eda'in i for i in targets]):
        env_eda()
    if any(['real'in i for i in targets]):
        env_data()
    if 'data-test' in targets:
        fetch_submissions(**TESTPARAMS)
        submissions_detail(TESTDIR)
        comments_detail(TESTDIR)
    if 'data-eda' in targets:
        fetch_submissions(**EDAPARAMS)
        submissions_detail(EDADIR)
        comments_detail(EDADIR)
    if 'data-real' in targets:
        fetch_submissions(**DATAPARAMS)
        submissions_detail(DATADIR)
        comments_detail(DATADIR)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)