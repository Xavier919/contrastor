import torch.nn as nn
import torch.nn.functional as F
from modules.utils import euclid_dis
import torch

class BaseNet1D(nn.Module):
    def __init__(self, input_channels=300, in_features=625, out_features=32):
        super(BaseNet1D, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = self.conv_block(input_channels, 250, k=5)
        self.conv2 = self.conv_block(250, 200, k=5)
        self.conv3 = self.conv_block(200, 150, k=5)
        self.conv4 = self.conv_block(150, 100, k=5)
        self.conv5 = self.conv_block(100, 50, k=5)
        self.convf = self.final_block(50, 1)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)


    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.convf(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels, k=5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
        )
        return block
        
    @staticmethod
    def final_block(in_channels, out_channels, k=1):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
        )
        return block

class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = euclid_dis((processed_a, processed_b))
        return distance