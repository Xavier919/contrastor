import pandas as pd
import numpy as np
import random
import torch
from nltk.tokenize import word_tokenize

def build_dataset(path, num_samples=-1, rnd_state=42):
    df1 = pd.read_json(path + "/fevrier.json")
    df2 = pd.read_json(path + "/janvier.json")
    df3 = pd.read_json(path + "/mars.json")
    df4 = pd.read_json(path + "/decembre.json")
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = df.dropna(subset=['text'])    
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
        
        if stacked_vectors.shape[1] < max_len:
            total_padding = max_len - stacked_vectors.shape[1]
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before
            padded_array = np.pad(stacked_vectors, ((0, 0), (padding_before, padding_after)), 'constant')
        elif stacked_vectors.shape[1] > max_len:
            excess_length = stacked_vectors.shape[1] - max_len
            trim_before = excess_length // 2
            trim_after = excess_length - trim_before
            padded_array = stacked_vectors[:, trim_before:-trim_after]
        else:
            padded_array = stacked_vectors
    
    return padded_array

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
    sum_square = torch.sum(torch.square(x - y), axis=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(torch.finfo(float).eps).to(sum_square.device)))

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.maximum(margin - y_pred, torch.tensor(0.0)))
    return torch.mean(y_true * square_pred + (1 - y_true) * margin_square)

def calculate_accuracy(y_pred, y_true):
    pred_labels = (y_pred < 0.5).float()  
    correct = (pred_labels == y_true).float()  
    accuracy = correct.mean()  
    return accuracy

def preprocess_text(text, language="french"):
    return word_tokenize(text.lower(), language=language)

def text_to_word2vec(text, model, max_len=10000):
    words = preprocess_text(text)
    vectors = [model[word] for word in words if word in model]
    
    if len(vectors) == 0:
        padded_array = np.zeros((model.vector_size, max_len))
    else:
        stacked_vectors = np.stack(vectors).T
        
        if stacked_vectors.shape[1] < max_len:
            total_padding = max_len - stacked_vectors.shape[1]
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before
            padded_array = np.pad(stacked_vectors, ((0, 0), (padding_before, padding_after)), 'constant')
        elif stacked_vectors.shape[1] > max_len:
            excess_length = stacked_vectors.shape[1] - max_len
            trim_before = excess_length // 2
            trim_after = excess_length - trim_before
            padded_array = stacked_vectors[:, trim_before:-trim_after]
        else:
            padded_array = stacked_vectors
    
    return padded_array

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for (data_a, data_b), target in dataloader:
        data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b)  
        loss = contrastive_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for (data_a, data_b), target in dataloader:
            data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
            output = model(data_a, data_b)
            accuracy_ = calculate_accuracy(output, target)
            total_accuracy += accuracy_.item() * data_a.size(0)  
            total_samples += data_a.size(0)

    return total_accuracy / total_samples