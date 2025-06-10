import torch


def eval(model, dataloader, criterion, device=None):
    with torch.no_grad():
        model.eval()
        accuracies = []
        losses = []
        for x, y in dataloader:
            x, y = x.to(device), y.squeeze().to(device)
            output = torch.argmax(model(x), dim=1)
            loss = criterion(output, y)
            correct = torch.sum(output == y)
            accuracies.append(correct/len(x))
            losses.append(loss.item())
        accuracies = torch.Tensor(accuracies)
        losses = torch.Tensor(losses)
        print(f"Val Accuracy: {accuracies.mean():.2f}%")
        print(f"Val Loss: {losses.mean():.2f}%")
