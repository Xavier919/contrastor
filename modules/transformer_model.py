import torch
import torch.nn as nn
import math

class BaseNetTransformer(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=1, n_heads=1, out_features=32):
        super(BaseNetTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, out_features)

    def forward(self, x):
        x = x.transpose(1, 2)  
        transformer_out = self.transformer_encoder(x)
        out = transformer_out.mean(dim=1)
        out = self.fc(out)
        return out

class SiameseTransformer(nn.Module):
    def __init__(self, base_network):
        super(SiameseTransformer, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = torch.norm(processed_a - processed_b, p=2, dim=1)
        return distance


class BaseNetTransformer(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=150, num_layers=1, n_heads=1, out_features=32, max_seq_length=5000):
        super(BaseNetTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        self.positional_encoding = self.create_positional_encoding(max_seq_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, out_features)

    def forward(self, x):
        seq_length = x.size(1)
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"The sequence length ({seq_length}) exceeds the maximum allowed length ({self.max_seq_length}).")

        positional_encoding = self.positional_encoding[:seq_length, :].to(x.device)
        x = x + positional_encoding

        transformer_out = self.transformer_encoder(x)
        
        out = transformer_out.mean(dim=1)
        
        out = self.fc(out)
        
        return out
    
    def create_positional_encoding(self, max_seq_length, embedding_dim):
        positional_encoding = torch.zeros(max_seq_length, embedding_dim)
        
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                if (i + 1) < embedding_dim:
                    positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))

        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding
    
class SiameseTransformer(nn.Module):
    def __init__(self, base_network):
        super(SiameseTransformer, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = torch.norm(processed_a - processed_b, p=2, dim=1)
        return distance