import os
import tensorflow as tf
import pandas as pd
import re
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import pickle
import numpy as np
import glob
import zipfile
from tensorflow.python.client import device_lib
from tqdm import tqdm
import shutil
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def load_nlp(source, path):
    print(device_lib.list_local_devices())
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(path, 'interim', 'label'))
    model_path = os.path.join(path, 'interim', 'label', 'nlp_model', 'model.json')
    weight_path = os.path.join(path, 'interim', 'label', 'nlp_model', 'model.h5')
    tokenizer_path = os.path.join(path, 'interim', 'label', 'nlp_model', 'tokenizer.pickle')
    model = load_model(model_path, weight_path)
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer

def preprocess_text(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def get_csvs(path):
    posts = glob.glob(os.path.join(path, '*.csv'))
    return pd.concat([pd.read_csv(i) for i in posts], ignore_index = True)

def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_model(json_path, weight_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_path)
    return loaded_model

def label(text, tokenizer, model, maxlen):
    tokenized = tokenizer.texts_to_sequences(text)
    padded_posts = keras.preprocessing.sequence.pad_sequences(tokenized, padding='post', maxlen=maxlen)
    predictions = model.predict(padded_posts, verbose = 0)
    return predictions

def label_comment(c_path, model, tokenizer, outpath, maxlen):
    comments = pd.read_csv(c_path)
    valid_comments = comments[(comments.body.notnull())&(comments.body!='[deleted]')&(comments.body!='[removed]')][['id','body','link_id']]
    predictions = label(valid_comments.body, tokenizer, model, maxlen)

    valid_predictions = pd.DataFrame(predictions, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    df = valid_predictions
    df['comment_id'] = valid_comments.reset_index().id
    df['post_id']=valid_comments.reset_index().link_id
    df['post_id']=df['post_id'].str[3:]
    df = df[['comment_id', 'post_id', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    deleted_comments = comments[(comments.body=='[deleted]')|(comments.body=='[removed]')][['id','link_id']]
    deleted_comments.columns = ['comment_id', 'post_id']
    deleted_comments['post_id']=deleted_comments['post_id'].str[3:]
    for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
        deleted_comments[c] = -1
    df = pd.concat([df, deleted_comments], ignore_index=True)
    df.to_csv(os.path.join(outpath, c_path.split('/')[-1]), index = False)

def label_comments(path, model, tokenizer, maxlen = 200, overlap = True):
    comment_path = os.path.join(path, 'raw', 'comments')
    outpath = os.path.join(path, 'interim', 'label', 'comment')
    comments_list = glob.glob(comment_path+'/*.csv')
    if overlap:
        print('labeling comments with overlapping')
        for c in tqdm(comments_list):
            label_comment(c, model, tokenizer, outpath, maxlen)
    else:
        print('labeling comments without overlapping')
        out_list = [s.split('/')[-1] for s in glob.glob(outpath+'/*.csv')]
        for c in tqdm(comments_list):
            if c.split('/')[-1] not in out_list:
                label_comment(c, model, tokenizer, outpath, maxlen)

def label_posts(path, model, tokenizer, thres = 0.5, maxlen = 200):
    print('labeling posts')
    post_path = os.path.join(path, 'raw', 'posts', '*.csv')
    outpath = os.path.join(path, 'interim', 'label', 'post')
    posts = pd.concat([pd.read_csv(i, usecols = ['id', 'selftext']) for i in glob.glob(post_path)], ignore_index=True)
    valid_posts = posts[(posts.selftext.notnull())&(posts.selftext!='[deleted]')&(posts.selftext!= '[removed]')]
    predictions = label(valid_posts.selftext.apply(preprocess_text), tokenizer, model, maxlen)
    valid_predictions = pd.DataFrame(predictions, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    valid_predictions['post_id'] = valid_posts['id'].copy().values
    valid_predictions = valid_predictions[['post_id', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    #valid_predictions.hist(bins = 20)
    deleted_posts = posts[(posts.selftext=='[deleted]')|(posts.selftext=='[removed]')][['id']]
    deleted_posts.columns = ['post_id']
    for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
        deleted_posts[c] = -1

    df = pd.concat([valid_predictions, deleted_posts], ignore_index= True)
    df.to_csv(os.path.join(outpath, 'post_sentimental.csv'), index = False)
    shutil.rmtree(os.path.join(path, 'interim', 'label', 'nlp_model'))

