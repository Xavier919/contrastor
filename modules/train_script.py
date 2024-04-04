import argparse
import torch
from modules.preprocess import text_edit
from modules.utils import *
from gensim.models import KeyedVectors
from modules.dataloader import PairedWord2VecDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import optim, nn
from torch.nn.parallel import DataParallel
from modules.transformer_model import BaseNetTransformer, SiameseTransformer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("split", type=int)
args = parser.parse_args()

if __name__ == "__main__":

    dataset = build_dataset(path="siamese_net/data",num_samples=args.num_samples, rnd_state=10)

    dataset = text_edit(dataset,grp_num=False,rm_newline=True,rm_punctuation=True,lowercase=True,lemmatize=False,html_=True,expand=True)

    X = np.array([x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])
    Y = np.array([x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    num_train_samples = len(X_train)*2
    num_test_samples = len(X_test)*2

    train_dataset = PairedWord2VecDataset(X_train, Y_train, text_to_word2vec, word2vec_model, num_train_samples)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16)

    test_dataset = PairedWord2VecDataset(X_test, Y_test, text_to_word2vec, word2vec_model, num_test_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16)

    base_net = BaseNetTransformer(embedding_dim=300, hidden_dim=128, num_layers=1, out_features=32)
    siamese_model = SiameseTransformer(base_net)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        siamese_model = nn.DataParallel(siamese_model)

    siamese_model = siamese_model.to(device)

    optimizer = optim.RMSprop(siamese_model.parameters(), lr=args.lr)

    epochs = args.epochs
    best_accuracy = 0
    for epoch in range(epochs):
        train_loss = train_epoch(siamese_model, train_loader, optimizer, device, epoch)
        val_accuracy = eval_model(siamese_model, train_loader, device, epoch)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if isinstance(siamese_model, nn.DataParallel):
                torch.save(siamese_model.module.state_dict(), f'best_model_{args.split}.pth')
                torch.save(siamese_model.module.base_network.state_dict(), f'base_net_model_{args.split}.pth')
            else:
                torch.save(siamese_model.state_dict(), f'best_model_{args.split}.pth')
                torch.save(siamese_model.base_network.state_dict(), f'base_net_model_{args.split}.pth')
            print("Model and Base Model saved as best model")
