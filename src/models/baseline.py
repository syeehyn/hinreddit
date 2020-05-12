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
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def get_csvs(dir_path):
    posts = glob.glob(dir_path+'*.csv')
    return pd.concat([pd.read_csv(i) for i in posts], ignore_index = True)

def clean_tweet(tweet): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

def sensitive_word(text):
    sen_lst = ['fuck', 'nigger', 'fucking', 'shit', 'sex', 'ass', 'asshole']
    lst = text.split()
    if any(word in lst for word in sen_lst):
        return 1
    else:
        return 0

def extract_feat(p_dir, l_path):
    
    posts = get_csvs(p_dir)
    labels = pd.read_csv(l_path)
    post_labeldf = pd.merge(labels,posts,left_on = 'post_id',right_on = 'id',how = 'inner').drop_duplicates()
    hate_benign_post = post_labeldf[(post_labeldf.label == 1) | (post_labeldf.label == 0)]
    hate_benign_post['selftext'] = hate_benign_post.selftext.fillna("")
    
    hate_benign_post['selftext'] = hate_benign_post['selftext'].apply(clean_tweet)
    hate_benign_post['len_content'] = hate_benign_post.selftext.apply(lambda x:len(x.split(" ")))
    hate_benign_post['sensitive'] = hate_benign_post.selftext.apply(sensitive_word)


    hate_benign_post['label']=hate_benign_post['label'].astype(int)
    return hate_benign_post[['num_comments', 'subreddit', 'score', 'len_content', 'sensitive', 'label']]

def preprocess(X, cat_feat = ['subreddit', 'sensitive'], num_feat = ['num_comments','len_content','score']):
    """
    provide the column transformer for the dataframe of simple features
    
    Args:
        X - dataframe of the apps w/ simple feature to create column transformer
        
    """
    cat_trans = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ])

    num_trans = Pipeline(steps = [('standard_scalar',StandardScaler())])

    return ColumnTransformer(transformers=[('cat', cat_trans,cat_feat), ('num', num_trans, num_feat)])

def calc_auc(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    return auc(fpr, tpr)

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
                       ('clf', LogisticRegression(class_weight='balanced'))
                       ])
    pipe.fit(X,y)
    X_te = df_test.drop(y_column, 1)
    y_te = df_test[y_column]
    y_pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    auc = calc_auc(pipe, X_te, y_te)
    return [tn, fp, fn, tp, auc]

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
                       ('clf', RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced'))
                       ])
    pipe.fit(X,y)
    X_te = df_test.drop(y_column, 1)
    y_te = df_test[y_column]
    y_pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    auc = calc_auc(pipe, X_te, y_te)
    return [tn, fp, fn, tp, auc]

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
    auc = calc_auc(pipe, X_te, y_te)
    return [tn, fp, fn, tp, auc]
def compute_metrics(mat):
    """
    output metrics including precision, recall, and AUC

    Args:
       mat - confustion matrix 
        
    """
    tn, fp, fn, tp = mat[0], mat[1], mat[2], mat[3]
    return tp/(tp+fp), tp/(tp+fn), mat[4]

def save_baseline_result(lr, rf, gbt):
    """
    given results of the baseline models, save them to file
    
    Args:
        lr - test result of logistic regression
        rf - test result of random forest
        gbt - test result of gradient boost classifier
        
    """
    baseline_result = pd.DataFrame([lr, rf, gbt], columns=['precision', 'recall', 'AUC'], index = np.array(['logistic regression', 'random forest', 'gradient boost']))
    #baseline_result.to_csv(os.path.join('output', 'baseline_result.csv'))
    return baseline_result
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
    lr = compute_metrics(result_lr)

    result_rf = result_RF(df_train, df_test, pre)
    rf = compute_metrics(result_rf)

    result_gbt = result_GBT(df_train, df_test, pre)
    gbt = compute_metrics(result_gbt)
    print('finish training baseline models')

    baseline_result = save_baseline_result(lr, rf, gbt)
    print('baseline test result saved to output directory')
    return baseline_result