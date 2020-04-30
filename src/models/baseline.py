import re
import glob, os, shutil
import gzip
import numpy as np
import pandas as pd
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def get_csvs(path):
    posts = glob.glob(path+'*.csv')
    return pd.concat([pd.read_csv(i) for i in posts], ignore_index = True)


def extract_feat(posts, labels):
    posts= posts.merge(labels, on='id')

    posts['length']=posts.selftext.str.split().str.len()

    posts['title_length']=posts.title.str.split().str.len()

    posts = posts[~(posts['label'].isna())&~(posts['label']==-1.0)]

    posts['label']=posts['label'].astype(int)
    return posts[['num_comments', 'subreddit', 'score', 'length', 'title_length', 'label']]

def preprocess(X):
    """
    provide the column transformer for the dataframe of simple features
    
    Args:
        X - dataframe of the apps w/ simple feature to create column transformer
        
    """
    cat_feat = ['subreddit']
    cat_trans = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ])

    num_trans = Pipeline(steps = [('standard_scalar',StandardScaler())])
    num_feat = ['num_comments','length', 'title_length','score']

    return ColumnTransformer(transformers=[('cat', cat_trans,cat_feat), ('num', num_trans, num_feat)])

def result_LR(df_train, df_test, pre, y_column = 'label'):
    """
    output the testing confusion matrix after feeding simple features into logistic regression models
    
    Args:
        df_train - dataframe for training set
        df_test - dataframe for test set
        pre - column transformer
        y_column - the column name of labels, default malware
        
    """
    X = df_train.drop(y_column, 1)
    y = df_train[y_column]
    pipe = Pipeline(steps=[('preprocessor', pre),
                       ('clf', LogisticRegression())
                       ])
    pipe.fit(X,y)
    X_te = df_test.drop(y_column, 1)
    y_te = df_test[y_column]
    y_pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    return tn, fp, fn, tp

def result_RF(df_train, df_test, pre, y_column = 'label'):
    """
    output the testing confusion matrix after feeding simple features into random forest models
    
    Args:
        df_train - dataframe for training set
        df_test - dataframe for test set
        pre - column transformer
        y_column - the column name of labels, default malware
        
    """
    X = df_train.drop(y_column, 1)
    y = df_train[y_column]
    pipe = Pipeline(steps=[('preprocessor', pre),
                       ('clf', RandomForestClassifier(max_depth=2, random_state=0))
                       ])
    pipe.fit(X,y)
    X_te = df_test.drop(y_column, 1)
    y_te = df_test[y_column]
    y_pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    return tn, fp, fn, tp

def result_GBT(df_train, df_test, pre, y_column = 'label'):
    """
    output the testing confusion matrix after feeding simple features into gradient boost classifier models
    
    Args:
        df_train - dataframe for training set
        df_test - dataframe for test set
        pre - column transformer
        y_column - the column name of labels, default malware
        
    """
    X = df_train.drop(y_column, 1)
    y = df_train[y_column]
    pipe = Pipeline(steps=[('preprocessor', pre),
                       ('clf', GradientBoostingClassifier())
                       ])
    pipe.fit(X,y)
    X_te = df_test.drop(y_column, 1)
    y_te = df_test[y_column]
    y_pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    return tn, fp, fn, tp
def compute_metrics(mat):
    """
    output metrics including 'tn', 'fp', 'fn', 'tp', 'acc', 'fnr'

    Args:
       mat - confustion matrix 
        
    """
    return mat + [(mat[0]+mat[3])/sum(mat), mat[2]/(mat[2]+mat[3])]#include confusion matrix and acc, fpr

def save_baseline_result(lr, rf, gbt):
    """
    given results of the baseline models, save them to file
    
    Args:
        lr - test result of logistic regression
        rf - test result of random forest
        gbt - test result of gradient boost classifier
        
    """
    baseline_result = pd.DataFrame([lr, rf, gbt], columns=['tn', 'fp', 'fn', 'tp', 'acc', 'fnr'], index = np.array(['logistic regression', 'random forest', 'gradient boost']))
    baseline_result.to_csv(os.path.join('output', 'baseline_result.csv'))
def baseline_model(df, y_col = 'label', test_size=0.3):
    """
    the whole process of training baseline model to saveing the result to file
    
    Args:
        df - dataframe of simple features
        y_col - column name for labels, default malware
        test_size - test size for train-test split, default 0.33
        
    """
    X = df.drop(y_col, 1)
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    print('preprocessing data...')
    pre = preprocess(X_train)

    df_train = X_train.assign(label = y_train)
    df_test = X_test.assign(label= y_test)
    
    print('start training baseline models...')
    result_lr = result_LR(df_train, df_test, pre)
    lr = compute_metrics(list(result_lr))

    result_rf = result_RF(df_train, df_test, pre)
    rf = compute_metrics(list(result_rf))

    result_gbt = result_GBT(df_train, df_test, pre)
    gbt = compute_metrics(list(result_gbt))
    print('finish training baseline models')

    baseline_result = pd.DataFrame([lr, rf, gbt], columns=['tn', 'fp', 'fn', 'tp', 'acc', 'fnr'], index = np.array(['logistic regression', 'random forest', 'gradient boost']))
    save_baseline_result(lr, rf, gbt)
    print('baseline test result saved to output directory')
    return baseline_result