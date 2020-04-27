from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow import keras
import numpy as np
import glob
def preprocess_text(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def embed_matrix(pretained):
    embeddings_dictionary = dict()

    glove_file = open(pretrained, encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            

def train_model(train_data, pretrained_text='src/glove.6B.100d.txt', maxlen = 200):
    toxic_comments = pd.read_csv(train_data)
    toxic_comments['text'] = toxic_comments.comment_text.apply(preprocess_text)
    toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    
    X = toxic_comments['text']
    y = toxic_comments_labels.values
    
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X = tokenizer.texts_to_sequences(X)
    vocab_size = len(tokenizer.word_index) + 1
    X = keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=maxlen)
    
    embedding_matrix = embed_matrix(pretrained_text)
    
    deep_inputs = keras.layers.Input(shape=(maxlen,))
    embedding_layer = keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = keras.layers.LSTM(128)(embedding_layer)
    dense_layer_1 = keras.layers.Dense(6, activation='sigmoid')(LSTM_Layer_1)
    model = keras.models.Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(X, y, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
    return tokenizer, model

    
def label_post(file, tokenizer, model, maxlen, outpath):
    posts = pd.read_csv(file)
    valid_posts = posts.selftext.replace('[deleted]', np.nan).replace('[removed]', np.nan).dropna().apply(preprocess_text)
    tokenized = tokenizer.texts_to_sequences(valid_posts)



    padded_posts = keras.preprocessing.sequence.pad_sequences(tokenized, padding='post', maxlen=maxlen)

    predictions = model.predict(padded_posts, verbose = 0)

    valid_predictions = pd.DataFrame(predictions, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

    #valid_predictions.hist(bins = 20)
    posts['label'] = 0

    posts.loc[posts.selftext.isin(['[deleted]', '[removed]']),'label'] = -1
    posts.loc[posts.selftext.isna(),'label'] = np.nan

    df = valid_predictions
    df['label']=0
    df.loc[(df.severe_toxic > 0.5)|(df.threat > 0.5)|(df.insult > 0.5)|(df.identity_hate > 0.5), 'label']=1


    df.label.index = posts.loc[~posts.selftext.isin(['[deleted]', '[removed]', np.nan])].index.values

    posts.loc[~posts.selftext.isin(['[deleted]', '[removed]', np.nan]),'label']=df.label

    
    posts[['id','label']].to_csv(outpath+file.split('/')[-1])
    
def label_posts(post_path, tokenizer, model, outpath, maxlen=200):
    for file in glob.glob(post_path+'*.csv'):
        label_post(file, tokenizer, model, maxlen, outpath)