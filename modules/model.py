import torch.nn as nn
import torch.nn.functional as F
from modules.utils import euclid_dis

class BaseNet1D(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(BaseNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)  
        self.avg_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        reduced_sequence_length = sequence_length // 4  
        fc_input_features = 64 * reduced_sequence_length

        self.fc1 = nn.Linear(fc_input_features, 128) 
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avg_pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

        
class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = euclid_dis((processed_a, processed_b))
        return distance