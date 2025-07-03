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


class GenericCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=11):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4**3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
