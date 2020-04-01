import praw as pr
import pandas as pd
from src import *
reddit = pr.Reddit(client_id=CLIENTID, \
                   client_secret=CLIENTSECRET, \
                   user_agent=USERAGENT,
                   username=USERNAME,
                  password=PASSWORD)
subreddits = [i.display_name for i in reddit.subreddits.default()]
def fetch_posts(num, subreddit):
    posts = []
    sub = reddit.subreddit(subreddit).random_rising(limit = num)
    for post in sub:
        posts.append([post.id, post.author,post.title, post.score, 
        post.subreddit, post.shortlink, post.num_comments, 
        post.selftext, post.created_utc, int(post.over_18), post.url])
    posts = pd.DataFrame(posts,columns=['post_id', 'author_name','title', 'score', 'subreddit', \
                                        'post_link', 'num_comments', 'body', 'created', 'over_18', 'content_url'])
    return posts

def _fetch_comment_detail(comment):
    try:
        res = {
                'author_id': comment.author.id,
                'created_utc': comment.created_utc,
                'score': comment.score
                }
    except AttributeError:
        try: res = {
                'author_id': None,
                'created_utc': comment.created_utc,
                'score': comment.score,
                'is_submitter': comment.is_submitter
                }
        except AttributeError:
            return
    if len(comment.replies.list()) == 0:
        res['replies'] = None
    else:
        res['replies'] = [_fetch_comment_detail(reply) for reply in comment.replies.list()] 
    return res

def fetch_post_detail(post):
    try:
        res = {
            'post_id': post.id,
            'author_id': post.author.id,
            'created_utc': post.created_utc,
            'score': post.score,
            'subreddit_id': post.subreddit.id,
            'upvote_ratio': post.upvote_ratio
            }
    except AttributeError:
        res = {
            'post_id': post.id,
            'author_id': None,
            'created_utc': post.created_utc,
            'score': post.score,
            'subreddit_id': post.subreddit.id,
            'upvote_ratio': post.upvote_ratio
            }
    comments = post.comments.list()
    res['comments'] = [_fetch_comment_detail(comment) for comment in comments]
    return res