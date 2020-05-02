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
LABL_DIR = os.path.join('interim', 'label', '*.csv')
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

def _process_nodes(posts, comm, labels):
    df_comm = comm.select('post_id', 'author', 'is_submitter')
    df_comm = df_comm.withColumn('is_post', F.lit(0))
    df_posts = posts.select('post_id', 'author', 'subreddit')
    df_posts = df_posts.withColumn('is_submitter', F.lit(1))
    df_posts = df_posts.withColumn('is_post', F.lit(1))
    df_comm = df_comm.join(df_posts, on = ['post_id'], how = 'inner')
    df_comm = df_comm.join(labels, on = ['post_id'], how = 'inner')
    df_posts = df_posts.join(labels, on = ['post_id'], how = 'inner')
    user_nodes_comm = df_comm.select(F.col('author').alias('node_name'),
                                F.col('post_id').alias('post_id'),
                                F.col('is_submitter'),
                                    'subreddit',
                                    'is_post',
                                    'label')
    user_nodes_post = df_posts.select(F.col('author').alias('node_name'),
                                    F.col('post_id').alias('post_id'),
                                    F.col('is_submitter'),
                                        'subreddit',
                                        'is_post',
                                        'label')
    user_nodes = user_nodes_comm.union(user_nodes_post)
    post_nodes = df_posts.select(F.col('post_id').alias('node_name'),
                                    F.col('post_id').alias('post_id'),
                                    F.col('is_submitter'),
                                    'subreddit',
                                    'is_post',
                                    'label')
    nodes = user_nodes.select('node_name').union(post_nodes.select('node_name')).dropDuplicates(['node_id']).dropna()
    stringIndexer = M.feature.StringIndexer(inputCol='node_name', outputCol='node_id')
    model = stringIndexer.setHandleInvalid("skip").fit(nodes)
    user_nodes = model.transform(user_nodes)
    post_nodes = model.transform(post_nodes)
    post_nodes = post_nodes.select('node_id', 'post_id','is_submitter', 'post_id', 'is_post', 'subreddit', 'label')
    user_nodes = model.transform(user_nodes.select(F.col('node_id').alias('tmp'), 
                F.col('post_id').alias('node_name'), 
                F.col('is_submitter'),
                F.col('post_id'),
                F.col('is_post'),
                'subreddit',
                'label'))
    user_nodes = user_nodes.select(F.col('tmp').alias('node_id'),
                                F.col('node_name').alias('post_id'),
                                F.col('is_submitter'),
                                F.col('is_post'),
                                F.col('node_id').alias('parent_id'),
                                'subreddit',
                                'label')
    nodes = user_nodes.select('node_id', 'post_id', 'is_submitter', 'is_post', 'subreddit', 'label')\
            .union(post_nodes.select('node_id', 'post_id', 'is_submitter','is_post', 'subreddit', 'label')).dropDuplicates(['node_id'])
    return model, nodes, user_nodes

def create_graph(fp):
    spark = _sparkSession()
    posts, comm, labels = _get_dfs(spark, fp)
    map_model, nodes, user_nodes = _process_nodes(posts, comm, labels)
    nodes.write.csv(os.path.join(fp, OUT_DIR, 'nodes.csv'), header = True)
    user_nodes.select('node_id', 'parent_id').distinct().write.csv(os.path.join(fp, OUT_DIR, 'edges.csv'), header = True)