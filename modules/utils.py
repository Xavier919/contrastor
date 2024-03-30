import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import random
import tensorflow as tf

def build_dataset(path, num_samples=-1, max_len=20000, rnd_state=42):
    df1 = pd.read_json(path + "/fevrier.json")
    df2 = pd.read_json(path + "/janvier.json")
    df3 = pd.read_json(path + "/mars.json")
    df4 = pd.read_json(path + "/decembre.json")
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = df.dropna(subset=['text'])
    
    df = df[df['text'].apply(lambda x: 50 <= len(x) <= max_len)]
    
    df['section_label'], _ = pd.factorize(df['section_1'])
    if num_samples != -1:
        df = df.sample(n=min(len(df), num_samples), replace=False, random_state=rnd_state)
    return df.T.to_dict()

def preprocess_text(text, language="french"):
    return word_tokenize(text.lower(), language=language)

def text_to_word2vec(text, model, max_len=20000):
    words = preprocess_text(text)
    vectors = [model[word] for word in words if word in model]
    
    if len(vectors) == 0:
        padded_array = np.zeros((model.vector_size, max_len))
    else:
        stacked_vectors = np.stack(vectors).T
        total_padding = max(0, max_len - stacked_vectors.shape[1])
        padding_before = total_padding // 2
        padding_after = total_padding - padding_before
        padded_array = np.pad(stacked_vectors, ((0, 0), (padding_before, padding_after)), 'constant')
    
    return padded_array[:, :max_len]

def zip_set(X, Y, num_pairs):
    zipped_list = list(zip(X, Y))
    pairs = []
    labels = []
    num_pairs = num_pairs
    for _ in range(num_pairs):
        sample1, sample2 = random.sample(zipped_list, 2)
        pairs.append([sample1[0], sample2[0]])
        if sample1[1] == sample2[1]:
            labels.append(1)
        else:
            labels.append(0) 
    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def euclid_dis(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    pred = tf.cast(y_pred < 0.5, y_true.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, pred), tf.float32))