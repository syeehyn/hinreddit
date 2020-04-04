import ujson as json
import sys
from src.etl import fetch_submissions
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
    return
def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    if 'post-test' in targets:
        res = {'results': fetch_submissions(**TESTPARAMS)}
        with open(os.path.join(TESTDIR, 'raw', 'posts', 'ingestion_results.json'), 'w') as fp:
            json.dump(res, fp)
        return 
    if 'comment-test' in targets:
        # posts_id = pd.read_csv(os.path.join(TESTDIR, 'raw', 'posts.csv')).post_id.tolist()
        # p_map(postsdetail_writer, posts_id, os.path.join(TESTDIR, 'raw', 'posts_detail'))
        # for post in tqdm(posts_id):
        #     postsdetail_writer(post, os.path.join(TESTDIR, 'raw', 'posts_detail'))
        # Parallel(n_jobs = NUM_WORKER)(delayed(postsdetail_writer)(post, os.path.join(TESTDIR, 'raw', 'posts_detail')) for post in tqdm(posts_id))
        return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)