import torch
from torch.utils.data import Dataset
import random

class PairedWord2VecDataset(Dataset):
    def __init__(self, X, Y, text_to_word2vec, word2vec_model, num_pairs):
        self.X = X
        self.Y = Y
        self.text_to_word2vec = text_to_word2vec
        self.word2vec_model = word2vec_model
        self.num_pairs = num_pairs
        self.pairs, self.labels = self.create_pairs()
        
    def create_pairs(self):
        zipped_list = list(zip(self.X, self.Y))
        pairs = []
        labels = []
        
        for _ in range(self.num_pairs):
            sample1, sample2 = random.sample(zipped_list, 2)
            pairs.append((sample1[0], sample2[0]))  
            labels.append(int(sample1[1] == sample2[1]))
        
        return pairs, torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        w2v_sample1 = self.text_to_word2vec(pair[0], self.word2vec_model)
        w2v_sample2 = self.text_to_word2vec(pair[1], self.word2vec_model)
        w2v_sample1 = torch.tensor(w2v_sample1, dtype=torch.float32)
        w2v_sample2 = torch.tensor(w2v_sample2, dtype=torch.float32)
        label = self.labels[idx]
        return (w2v_sample1, w2v_sample2), label
