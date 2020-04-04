import praw as pr
import pandas as pd
from src import *
import ujson as json
import requests
import pandas as pd
import os
from os.path import join
from tqdm import tqdm
import time
from joblib import Parallel, delayed
from p_tqdm import p_umap
from glob import glob
API = 'https://api.pushshift.io/'
COMMENT = join(API, 'reddit/search/comment/')
SUBMISSION = join(API, 'reddit/search/submission/')
SUBMISSION_DETAIL = join(API, 'reddit/submission/comment_ids/')
POST_DIR = 'raw/posts'
POST_DETAIL_DIR = 'raw/posts_detail'
def fetch_post(subreddit, sort_type, sort, size, before, meta):
    params = '?' + 'subreddit=' + subreddit + \
             '&' + 'sort_type=' + sort_type + \
             '&' + 'sort=' + sort + \
             '&' + 'size=' + size + \
             '&' + 'before=' + before
    r = requests.get(join(SUBMISSION, params))
    if r.status_code == 200:
        try:
            data = pd.DataFrame(r.json()['data'])[meta]
            return data, str(data.created_utc.min())
        except KeyError:
            data = pd.DataFrame(r.json()['data'])
            return data, False
    else:
        time.sleep(2)
        r = requests.get(join(SUBMISSION, params))
        if r.status_code == 200:
            try:
                data = pd.DataFrame(r.json()['data'])[meta]
                return data, str(data.created_utc.min())
            except KeyError:
                data = pd.DataFrame(r.json()['data'])
                return data, False
        else:
            time.sleep(5)
            r = requests.get(join(SUBMISSION, params))
            if r.status_code == 200:
                try:
                    data = pd.DataFrame(r.json()['data'])[meta]
                    return data, str(data.created_utc.min())
                except KeyError:
                    data = pd.DataFrame(r.json()['data'])
                    return data, False
            else:
                return None
def fetch_posts(subreddit, total, meta, filepath, sort_type, sort, size, start):
    num_epoch = -(-int(total) // int(size))
    start_time = start
    for i in range(num_epoch):
        last_time = start_time
        try:
            process, start_time = fetch_post(subreddit, sort_type, sort, size, start_time, meta)
        # except TypeError:
        #     time.sleep(5)
        #     try: 
        #         process, start_time = fetch_post(subreddit, sort_type, sort, size, start_time, meta)
        #     except TypeError:
        #         try:
        #             time.sleep(5)
#             process, start_time = fetch_post(subreddit, sort_type, sort, size, start_time, meta)
        except TypeError:
            return {'subreddit': subreddit, 'result': 'unsuccess', 'status': i, 'last_time': last_time}
        if start_time != False:
            if not os.path.exists(join(filepath, POST_DIR, subreddit+'.csv')):
                process.to_csv(join(filepath, POST_DIR, subreddit+'.csv'), index = False)
            else:
                process.to_csv(join(filepath, POST_DIR, subreddit+'.csv'), index = False, mode='a', header = False)
        else:
            process.to_csv(join(filepath, POST_DIR, subreddit+'_failed.csv'), index = False)
            return {'subreddit': subreddit, 'result': 'unsuccess', 'status': i, 'last_time': last_time}
        time.sleep(.5)
    return {'subreddit': subreddit,'result': 'success', 'status': num_epoch, 'last_time': last_time}
def fetch_submissions(**kwargs):
    post_args, meta_args = kwargs['POST_ARGS'], kwargs['META_ARGS']
    filepath, total, meta, subreddits = meta_args['filepath'], meta_args['total'], \
                                        meta_args['meta'], meta_args['subreddits']
    sort_type, sort, size, start = post_args['sort_type'], post_args['sort'], post_args['size'], post_args['start']
    tolist = lambda x: [x for _ in range(len(subreddits))]
    res = p_umap(fetch_posts, subreddits, tolist(total), tolist(meta), tolist(filepath), tolist(sort_type), tolist(sort), tolist(size), tolist(start), num_cpus = 8)
    # res = Parallel(n_jobs = 8)(delayed(fetch_posts)\
    #                     (subreddit, total, meta, filepath, sort_type, sort, size, start) \
    #                     for subreddit in tqdm(subreddits))
    with open(os.path.join(filepath, 'raw', 'posts', 'log.json'), 'w') as fp:
            json.dump(res, fp)
    return res

def submission_detail(i):
    r = requests.get(join(SUBMISSION_DETAIL, i))
    if r.status_code == 200:
        return {'submission_id': i, 'comment_ids': r.json()['data']}
    else:
        time.sleep(2)
        r = requests.get(join(SUBMISSION_DETAIL, i))
        if r.status_code == 200:
            return {'submission_id': i, 'comment_ids': r.json()['data']}
        else:
            time.sleep(5)
            r = requests.get(join(SUBMISSION_DETAIL, i))
            if r.status_code == 200:
                return {'submission_id': i, 'comment_ids': r.json()['data']}
            else:
                return {'submission_id': i, 'comment_ids': []}
def submissions_detail(filepath):
    subreddits_fp = glob(join(filepath, POST_DIR, '*.csv'))
    subreddits = [i.split('/')[-1][:-4] for i in subreddits_fp]
    n, N = 1, len(subreddits)
    for subreddit, fp in zip(subreddits,subreddits_fp):
        print('fetching {0} subreddit details, Progress: {1}/{2}'.format(subreddit, str(n), str(N)))
        ids = pd.read_csv(fp).id.tolist()
        rest = p_umap(submission_detail, ids, num_cpus = 8)
        with open(join(filepath, POST_DETAIL_DIR, subreddit+'.json'), 'w') as fp:
            json.dump(rest, fp)
        n += 1