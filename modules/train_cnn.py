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
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("split", type=int)
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, X, Y, word2vec_model):
        word2vec_arrays = [text_to_word2vec(x, word2vec_model) for x in X]
        combined_array = np.array(word2vec_arrays)  # Combining into one numpy array
        self.X = torch.tensor(combined_array, dtype=torch.float32)  # More efficient tensor creation
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

writer = SummaryWriter()
test_writer = SummaryWriter()

if __name__ == "__main__":

    dataset = build_dataset(path="siamese_net/data", num_samples=args.num_samples, rnd_state=10)
    dataset = text_edit(dataset, grp_num=False, rm_newline=True, rm_punctuation=True, lowercase=True, lemmatize=False, html_=True, expand=False)

    X = np.array([x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])
    Y = np.array([x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset(X_train, Y_train, word2vec_model)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_dataset = CustomDataset(X_test, Y_test, word2vec_model)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = CNN_NLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Batch_size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Split: {args.split}")

    num_epochs = args.epochs
    best_loss = 10
    for epoch in range(num_epochs):
        model.train()
        train_losses = 0
        for X, Y in train_dataloader:
            X = X.view(len(X), 5000, 300).to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            train_losses += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch)
            torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}, Train loss: {train_losses / len(train_dataloader)}')

        model.eval()
        test_losses = 0
        for X, Y in test_dataloader:
            X = X.view(len(X), 5000, 300).to(device)
            Y = Y.to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)
            test_losses += loss.item()
            test_writer.add_scalar("Loss/test", loss.item(), epoch)
            torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}, Test loss: {test_losses / len(test_dataloader)}')

        if best_loss > (test_losses / len(test_dataloader)):
            best_loss = test_losses/len(test_dataloader)
            torch.save(model.state_dict(), f'best_cnn_{args.split}.pth')
            print("Model saved as best model")