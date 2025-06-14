import torch


def train(model, dataloader, epochs, optimizer, criterion, device=None):
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
