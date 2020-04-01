import json
import sys
from src.etl import fetch_posts, subreddits
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
    return
def main(targets):
    if any(['test'in i for i in targets]):
        env_test()
    if 'data-test' in targets:
        dfs = Parallel(n_jobs = NUM_WORKER)(delayed(fetch_posts)(TESTNUM, subreddit) for subreddit in tqdm(subreddits))
        dfs = pd.concat(dfs, ignore_index=True)
        dfs.to_csv(os.path.join(TESTDIR, 'raw', 'posts.csv'), index = False)
        return 


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)