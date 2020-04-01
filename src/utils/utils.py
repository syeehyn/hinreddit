import praw as pr
import pandas as pd
import os
import json
from src import *
from src.etl import fetch_post_detail
reddit = pr.Reddit(client_id=CLIENTID, \
                   client_secret=CLIENTSECRET, \
                   user_agent=USERAGENT,
                   username=USERNAME,
                  password=PASSWORD)
def get_submission(post_id):
    return reddit.submission(id = post_id)

def postsdetail_writer(post_id, fp):
    if os.path.exists(os.path.join(fp, post_id + '.json')):
        return
    res = fetch_post_detail(get_submission(post_id))
    with open(os.path.join(fp, post_id + '.json'), 'w') as fp:
        json.dump(res, fp)
    return