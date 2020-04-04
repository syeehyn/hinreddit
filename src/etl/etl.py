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
API = 'https://api.pushshift.io/'
COMMENT = join(API, 'reddit/search/comment/')
SUBMISSION = join(API, 'reddit/search/submission/')
COMMENT_DETAIL = join(API, 'reddit/submission/comment_ids/')

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
def fetch_posts(subreddit, total, meta, filepath,**kwargs):
    num_epoch = -(-int(total) // int(kwargs['size']))
    start_time = kwargs['start']
    for i in range(num_epoch):
        last_time = start_time
        try:
            process, start_time = fetch_post(subreddit, kwargs['sort_type'], kwargs['sort'], kwargs['size'], start_time, meta)
        except TypeError:
            time.sleep(5)
            try: 
                process, start_time = fetch_post(subreddit, kwargs['sort_type'], kwargs['sort'], kwargs['size'], start_time, meta)
            except TypeError:
                try:
                    time.sleep(5)
                    process, start_time = fetch_post(subreddit, kwargs['sort_type'], kwargs['sort'], kwargs['size'], start_time, meta)
                except TypeError:
                    return {subreddit: 'unsuccess', 'status': i, 'last_time': last_time}
        if start_time != False:
            if not os.path.exists(join(filepath, subreddit+'.csv')):
                process.to_csv(join(filepath, subreddit+'.csv'), index = False)
            else:
                process.to_csv(join(filepath, subreddit+'.csv'), index = False, mode='a', header = False)
        else:
            process.to_csv(join(filepath, subreddit+'_failed.csv'), index = False)
            return {subreddit: 'unsuccess', 'status': i, 'last_time': last_time}
        time.sleep(.5)
    return {subreddit: 'success', 'status': num_epoch, 'last_time': last_time}
def fetch_submissions(**kwargs):
    post_args, meta_args = kwargs['POST_ARGS'], kwargs['META_ARGS']
    filepath, total, meta, subreddits = meta_args['filepath'], meta_args['total'], \
                                        meta_args['meta'], meta_args['subreddits']
    res = Parallel(n_jobs = 8)(delayed(fetch_posts)\
                        (subreddit, total, meta, filepath, **post_args) \
                        for subreddit in tqdm(subreddits))
    return res