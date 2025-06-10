import torch.nn as nn
from torch.nn.functional import relu


class UNET3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, 2)
        self.conv2 = nn.Conv3d(32, 64, 2)
        self.max1 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(64, 128, 2)
        self.conv4 = nn.Conv3d(128, 256, 2)
        self.conv5 = nn.Conv3d(256, 256, 2)

        self.dense1 = nn.Linear(256*4*4*4, 128)
        self.dense2 = nn.Linear(128, 11)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.max1(relu(self.conv2(x)))
        x = relu(self.conv3(x))
        x = self.max1(relu(self.conv4(x)))
        x = self.conv5(x)
        x = self.dense1(x.view(x.shape[0], -1))
        x = self.dense2(x)
        return x
