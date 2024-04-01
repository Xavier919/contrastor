import argparse
import torch
from modules.preprocess import text_edit
from modules.utils import *
from gensim.models import KeyedVectors
from modules.dataloader import PairedWord2VecDataset
from modules.transformer_model import BaseNetTransformer, SiameseTransformer
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    dataset = build_dataset(path="siamese_net/data",num_samples=args.num_samples, rnd_state=10)

    dataset = text_edit(dataset,grp_num=False,rm_newline=True,rm_punctuation=True,lowercase=True,lemmatize=False,html_=True,expand=True)

    X = [x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires', 'debats']]
    Y = [x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires', 'debats']]

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    text = "Ceci est un texte exemple"
    vector = text_to_word2vec(text, word2vec_model)
    shape = vector.shape[0]

    base_net = BaseNetTransformer(embedding_dim=300, hidden_dim=64, num_layers=1, out_features=32)
    siamese_model = SiameseTransformer(base_net)

    model_path = "best_model.pth"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    siamese_model.load_state_dict(state_dict)

    siamese_model = siamese_model.to(device)

    siamese_model.eval()
    results = []
    for X_, Y_ in  list(zip(X,Y)):
        X_ = text_to_word2vec(X_, word2vec_model)
        X_ = torch.tensor(X_).to(device)
        output = siamese_model(X_).detach()
        results.append((output, Y_))

    pickle.dump(results, open('results.pkl', 'wb'))