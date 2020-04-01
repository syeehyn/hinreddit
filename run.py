import json
import sys
from src.etl import fetch_posts, subreddits, fetch_post_detail
from src.utils import postsdetail_writer
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import psutil
NUM_WORKER = psutil.cpu_count(logical = False)
TESTDIR = json.load(open('config/test-params.json'))['dir']
TESTNUM = json.load(open('config/test-params.json'))['num']
def env_test():
    if not os.path.exists(TESTDIR):
        os.mkdir(TESTDIR)
    if not os.path.exists(os.path.join(TESTDIR, 'raw')):
        os.mkdir(os.path.join(TESTDIR, 'raw'))
    if not os.path.exists(os.path.join(TESTDIR, 'raw', 'posts_detail')):
        os.mkdir(os.path.join(TESTDIR, 'raw', 'posts_detail'))
    return
def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    if 'post-test' in targets:
        df = Parallel(n_jobs = NUM_WORKER)(delayed(fetch_posts)(TESTNUM, subreddit) for subreddit in tqdm(subreddits))
        df = pd.concat(df, ignore_index=True)
        df.to_csv(os.path.join(TESTDIR, 'raw', 'posts.csv'), index = False)
        return 
    if 'comment-test' in targets:
        posts_id = pd.read_csv(os.path.join(TESTDIR, 'raw', 'posts.csv')).post_id
        # for post in tqdm(posts_id):
        #     postsdetail_writer(post, os.path.join(TESTDIR, 'raw', 'posts_detail'))
        Parallel(n_jobs = NUM_WORKER)(delayed(postsdetail_writer)(post, os.path.join(TESTDIR, 'raw', 'posts_detail')) for post in tqdm(posts_id))
        return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)