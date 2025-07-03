import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1e-5, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = net(images)
            loss = criterion(output, labels.long().squeeze(dim=1))
            loss.backward()
            optimizer.step()


def _train(model, dataloader, epochs, optimizer, criterion, device=None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(epochs):
        model.train()
        if epochs > 1:
            print(f"Epoch: {i+1}/{epochs}")
        for j, data in enumerate(dataloader):
            x, y = data
            optimizer.zero_grad()
            x, y = x.to(device), y.squeeze().to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f'Batch {j+1}\tLoss {loss.item()}')
