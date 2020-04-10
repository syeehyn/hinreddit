import json
import sys
from src.etl import fetch_submissions, submissions_detail, comments_detail
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import psutil
from p_tqdm import p_map
NUM_WORKER = psutil.cpu_count(logical = False)
TESTPARAMS = json.load(open('config/test-params.json'))
TESTDIR = os.path.join('./tests')
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
def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    if 'data-test' in targets:
        fetch_submissions(**TESTPARAMS)
        print('post fetched')
        submissions_detail(TESTDIR)
        print('post details fetched')
        comments_detail(TESTDIR)
        print('all fetched')
    # if 'post-test' in targets:
    #     fetch_submissions(**TESTPARAMS)
    #     return 'Done'
    # if 'comment-test' in targets:
    #     submissions_detail(TESTDIR)
    #     return 'Done'
    # if 'comment-detail-test' in targets:
    #     comments_detail(TESTDIR)
    #     return  'Done'


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)