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
