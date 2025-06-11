from medical_fl import UNET3D
import medmnist
from medmnist import INFO, Evaluator

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from medical_fl import train

data_flag = 'organmnist3d'
download = True


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).float()


NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001
size = 28  # Options: 28, 64

info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])
task = info['task']

DataClass = getattr(medmnist, info['python_class'])

# Train Data
train_dataset = DataClass(split='train', transform=Compose([ToTensor()]), download=download, size=size)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Val Data
val_dataset = DataClass(split='val', transform=Compose([ToTensor()]), download=download, size=size)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

# Test Data
test_dataset = DataClass(split='test', transform=Compose([ToTensor()]), download=download, size=size)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# Model parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNET3D().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for step in range(NUM_EPOCHS):
    print(f"Epoch: {step+1}")
    train(model, train_loader, 1, optimizer, criterion, device=device)
    eval(model, val_loader, criterion, device=device)

print("Testing Model")
eval(model, test_loader, criterion, device=device)
torch.save(model.state_dict(), '../models/UNET3D.pth')
