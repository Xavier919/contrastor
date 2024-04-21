import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.utils import *
from modules.preprocess import *
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from modules.cnn_model import CNN_NLP
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("split", type=int)
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, X, Y, word2vec_model):
        self.X = X  
        self.Y = torch.tensor(Y, dtype=torch.long)  
        self.word2vec_model = word2vec_model  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        word2vec_array = text_to_word2vec(self.X[idx], self.word2vec_model)
        x_tensor = torch.tensor(word2vec_array, dtype=torch.float32)
        return x_tensor, self.Y[idx]

if __name__ == "__main__":

    dataset = build_dataset(path="siamese_net/data", num_samples=args.num_samples, rnd_state=10)
    dataset = text_edit(dataset, grp_num=False, rm_newline=False, rm_punctuation=False, lowercase=False, lemmatize=False, html_=False, expand=False)

    X = np.array([x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])
    Y = np.array([x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = CustomDataset(X_test, Y_test, word2vec_model)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model = CNN_NLP()

    model_path = f"best_cnn_{args.split}.pth"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {key:value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)

    print(f"Split: {args.split}")

    softmax = nn.Softmax(dim=1)

    results = []
    model.eval()
    for X, Y in test_dataloader:
        X = X.view(len(X), 5000, 300).to(device)
        output = model(X).view(-1)
        output = np.argmax(softmax(output).cpu().detach())
        results.append((output.numpy(), Y.numpy()))

    pickle.dump(results, open(f'results_cnn_{args.split}.pkl', 'wb'))