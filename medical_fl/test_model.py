import torch.nn as nn
from typing import List


# 1. Definição do Modelo (uma CNN simples)
class UnetModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.module(x)


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, features: List[int] = [16, 64]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(UnetModule(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
            x = self.pool(x)

        return x
