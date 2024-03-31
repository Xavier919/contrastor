import torch.nn as nn
import torch.nn.functional as F
from modules.utils import euclid_dis

class BaseNet1D(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(BaseNet1D, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = self.conv_block(input_channels, 64, k=5)
        self.conv2 = self.conv_block(64, 32, k=5)
        self.conv3 = self.conv_block(32, 16, k=5)
        self.conv4 = self.conv_block(16, 8, k=5)
        self.dconvf = self.final_block(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        block1 = self.conv1(x)
        x = self.maxpool(block1)
        block2 = self.conv2(x) 
        x = self.maxpool(block2)
        block3 = self.conv3(x) 
        x = self.maxpool(block3)
        block4 = self.conv4(x) 
        x = self.maxpool(block4)
        x = self.dconvf(x)
        return self.sigmoid(x)

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