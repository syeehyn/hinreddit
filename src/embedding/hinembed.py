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
OUT_DIR = os.path.join('interim', 'hinembed')


def construct_matrices(datapath):
    ######### spark env
    SparkContext.setSystemProperty('spark.executor.memory', '64g')
    sc = SparkContext("local", "App Name")
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)
    spark.conf.set('spark.ui.showConsoleProgress', True)
    spark.conf.set("spark.sql.shuffle.partitions", NUM_WORKER)
    ######### defining terms and loading data
    COMM = os.path.join(datapath, COMM_DIR)
    LABL = os.path.join(datapath, LABL_DIR)
    labels = spark.read.format("csv").option("header", "true").load(LABL)
    labels = labels.select(F.col('id').alias('post_id'), 'label')
    comm = spark.read.format("csv").option("header", "true").load(COMM)
    comm = comm.select(F.col('id').alias('comment_id'), 'author', comm.link_id.substr(4, 10).alias('post_id'))
    df = comm.join(labels, 'post_id', 'left')
    df = df.where(~F.col('label').isNull()).where(F.col('label') != -1)
    ######### index converting
    stringIndexer = M.feature.StringIndexer(inputCol='post_id', outputCol='post_id_ind')
    model = stringIndexer.fit(df)
    df = model.transform(df)
    stringIndexer = M.feature.StringIndexer(inputCol='author', outputCol='author_id')
    model = stringIndexer.fit(df)
    df = model.transform(df)
    df.select('post_id', F.col('post_id_ind').cast('int'), 'label').dropDuplicates().toPandas().to_csv(os.path.join(datapath, OUT_DIR, 'A_post_ref.csv'), index = False)
    df.select('author', F.col('author_id').cast('int')).dropDuplicates().toPandas().to_csv(os.path.join(datapath, OUT_DIR, 'A_author_ref.csv'), index = False)
    ######### matrices constructing
    A_prec = df.select(F.col('post_id_ind').cast('int'), F.col('author_id').cast('int')).dropDuplicates()
    A = A_prec.toPandas().values.astype(int)
    num_post = np.unique(A[:, 0]).shape[0]
    num_author = np.unique(A[:, 1]).shape[0]
    values = np.full(shape=A.shape[0], fill_value=1, dtype='i1')
    A = sparse.coo_matrix(
                    (values, (A[:,0], A[:,1])), shape=(num_post, num_author)
        )
    A = (A > 0).astype(int)
    sparse.save_npz(os.path.join(datapath, OUT_DIR), A)
    del A
    print('finished constructing A')