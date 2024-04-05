import pandas as pd
import numpy as np
import random
import torch
from nltk.tokenize import word_tokenize
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import io
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from tqdm import tqdm

writer = SummaryWriter()
test_writer = SummaryWriter()

def build_dataset(path, num_samples=-1, rnd_state=42):
    df1 = pd.read_json(path + "/fevrier.json")
    df2 = pd.read_json(path + "/janvier.json")
    df3 = pd.read_json(path + "/mars.json")
    df4 = pd.read_json(path + "/decembre.json")
    df5 = pd.read_json(path + "/novembre.json")
    df6 = pd.read_json(path + "/aout.json")
    df7 = pd.read_json(path + "/septembre.json")
    df8 = pd.read_json(path + "/octobre.json")
    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8], ignore_index=True)
    df = df.dropna(subset=['text'])    
    df['section_label'], _ = pd.factorize(df['section_1'])
    if num_samples != -1:
        df = df.sample(n=min(len(df), num_samples), replace=False, random_state=rnd_state)
    return df.T.to_dict()

def preprocess_text(text, language="french"):
    return word_tokenize(text.lower(), language=language)

def text_to_word2vec(text, model, max_len=15000):
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

def euclid_dis(vects):
    x, y = vects
    x_flat = x.view(x.size(0), -1)  
    y_flat = y.view(y.size(0), -1)  
    sum_square = torch.sum(torch.square(x_flat - y_flat), axis=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(torch.finfo(float).eps).to(sum_square.device)))

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.clamp(margin - y_pred, min=0))
    loss = torch.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def calculate_accuracy(y_pred, y_true):
    pred_labels = (y_pred < 0.5).float()  
    correct = (pred_labels == y_true).float()  
    accuracy = correct.mean()  
    return accuracy

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for (data_a, data_b), target in tqdm(dataloader):
        data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b)  
        loss = contrastive_loss(target, output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), epoch)
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device, epoch):
    model.eval()
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for (data_a, data_b), target in dataloader:
            data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
            output = model(data_a, data_b)
            loss = contrastive_loss(target, output)
            test_writer.add_scalar("Loss/test", loss.item(), epoch)
            accuracy_ = calculate_accuracy(output, target)
            total_accuracy += accuracy_.item() * data_a.size(0)  
            total_samples += data_a.size(0)

    return total_accuracy / total_samples

def tune_logistic_regression(X_train, Y_train):
    model = LogisticRegression(multi_class="multinomial", max_iter=5000)
    param_grid = {
        "solver": ["newton-cg", "lbfgs", "saga"],
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model


def tune_svm(X_train, Y_train):
    model = SVC()
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4, 5, 6, 7, 8, 9],
        "gamma": ["scale", "auto"],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model


def tune_mlp(X_train, Y_train):
    model = MLPClassifier(max_iter=5000)
    param_grid = {
        "hidden_layer_sizes": [(25,), (50,)],
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model


def tune_naive_bayes(X_train, Y_train):
    model = MultinomialNB()
    param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0], "fit_prior": [True, False]}
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model


def evaluate(Y_test, y_pred):
    print("Precision: ",precision_score(Y_test, y_pred, average="weighted", zero_division=0)),
    print("Recall: ", recall_score(Y_test, y_pred, average="weighted", zero_division=0))
    print("F1_score: ", f1_score(Y_test, y_pred, average="weighted", zero_division=0))
    print("accuracy: ", accuracy_score(Y_test, y_pred))

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        

def get_data_splits(X, Y, split, n_splits=5, shuffle=True, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))  
    train_index, test_index = splits[split]  
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test