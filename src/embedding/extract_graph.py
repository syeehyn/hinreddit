import os
import pandas as pd
import numpy as np
from scipy import sparse
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T
from src import *
from glob import glob

COMM_DIR = os.path.join('raw', 'comments', '*.csv')
LABL_DIR = os.path.join('label', '*.csv')
POST_DIR = os.path.join('raw', 'posts', '*.csv')
OUT_DIR = os.path.join('interim', 'graph_table')

def _sparkSession():
    SparkContext.setSystemProperty('spark.executor.memory', '64g')
    sc = SparkContext("local", "App Name")
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)
    spark.conf.set('spark.ui.showConsoleProgress', True)
    spark.conf.set("spark.sql.shuffle.partitions", NUM_WORKER)
    return spark

def _get_dfs(spark, fp):
    LABL, POST, COMM = os.path.join(fp, LABL_DIR), os.path.join(fp, POST_DIR), os.path.join(fp, COMM_DIR)
    labels = spark.read.format("csv").option("header", "true").load(LABL)
    labels = labels.select(F.col('id').alias('post_id'), 'label')
    posts = pd.concat([pd.read_csv(i) for i in glob(POST)], ignore_index = True)
    posts = posts[['id', 'author', 'subreddit']]
    posts.author = posts.author.str.lower()
    posts.subreddit  = posts.subreddit.str.lower()
    posts.columns = ['post_id', 'author', 'subreddit']
    posts = spark.createDataFrame(posts)
    posts = posts.where(F.col('author') != 'automoderator')
    comm = spark.read.format("csv").option("header", "true").load(COMM)
    comm = comm.select(F.col('id').alias('comment_id'), \
                    F.lower(F.col('author')).alias('author'), \
                    comm.link_id.substr(4, 10).alias('post_id'), 
                    F.lower(F.col('subreddit')).alias('subreddit'),
                    (F.col('is_submitter')==True).cast('int').alias('is_submitter'))
    comm = comm.where(F.col('author') != 'automoderator')
    return posts, comm, labels

def _process_nodes(posts, comm):
    stringIndexer = M.feature.StringIndexer(inputCol='subreddit', outputCol='subreddit_')
    model_string = stringIndexer.setHandleInvalid("keep").fit(comm)
    td = model_string.transform(comm)
    onehot_comm = td.select('post_id', 'author', 'subreddit', 'subreddit_', 'is_submitter')
    onehot_comm = onehot_comm.withColumn('is_post', F.lit(0))
    td = model_string.transform(posts)
    onehot_posts = td.select('post_id', 'author', 'subreddit', 'subreddit_')
    onehot_posts = onehot_comm.withColumn('is_submitter', F.lit(1))
    onehot_posts = onehot_posts.withColumn('is_post', F.lit(1))
    user_nodes_comm = onehot_comm.select(F.col('author').alias('node_name'),
                                F.col('post_id').alias('post_id'),
                                F.col('is_submitter').alias('user_feature'),
                                F.col('subreddit_').alias('post_feature'),
                                    'is_post')
    user_nodes_post = onehot_comm.select(F.col('author').alias('node_name'),
                                    F.col('post_id').alias('post_id'),
                                    F.col('is_submitter').alias('user_feature'),
                                    F.col('subreddit_').alias('post_feature'),
                                        'is_post')
    user_nodes = user_nodes_comm.union(user_nodes_post)
    post_nodes = onehot_posts.select(F.col('post_id').alias('node_name'),
                                    F.col('is_submitter').alias('user_feature'),
                                    F.col('subreddit_').alias('post_feature'),
                                    'is_post')
    nodes = user_nodes.select('node_name').union(post_nodes.select('node_name')).dropna()
    stringIndexer = M.feature.StringIndexer(inputCol='node_name', outputCol='node_id')
    model = stringIndexer.setHandleInvalid("keep").fit(nodes)
    user_nodes = model.transform(user_nodes)
    post_nodes = model.transform(post_nodes)
    post_nodes = post_nodes.select('node_id', 'user_feature', 'post_feature', 'is_post')
    user_nodes = model.transform(user_nodes.select(F.col('node_id').alias('tmp'), 
                F.col('post_id').alias('node_name'), 
                F.col('user_feature'),
                F.col('post_feature'),
                F.col('is_post')))
    user_nodes = user_nodes.select(F.col('tmp').alias('node_id'),
                                F.col('user_feature'),
                                F.col('post_feature'),
                                F.col('is_post'),
                                F.col('node_id').alias('parent_id'))
    return model, post_nodes, user_nodes

def create_graph(fp):
    spark = _sparkSession()
    posts, comm, labels = _get_dfs(spark, fp)
    map_model, post_nodes, user_nodes = _process_nodes(posts, comm) #todo: labels
    nodes = user_nodes.select('node_id', 'user_feature', 'post_feature', 'is_post').union(post_nodes.select('node_id', 'user_feature', 'post_feature', 'is_post')).dropDuplicates(['node_id'])
    nodes.write.csv(os.path.join(fp, OUT_DIR, 'nodes.csv'))
    user_nodes.select('node_id', 'parent_id').distinct().write.csv(os.path.join(fp, OUT_DIR, 'edges.csv'))