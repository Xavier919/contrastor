import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetRNN(nn.Module):
    def __init__(self, input_features=300, hidden_dim=128, num_layers=2, out_features=32):
        super(BaseNetRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_step_output = lstm_out[:, -1, :]
        out = self.fc(last_step_output)
        return out

class SiameseRNN(nn.Module):
    def __init__(self, base_network):
        super(SiameseRNN, self).__init__()
        self.base_network = base_network

    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = torch.norm(processed_a - processed_b, p=2, dim=1)
        return distance
