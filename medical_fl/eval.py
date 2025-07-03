import torch
from typing import Dict, Optional, Tuple
from medical_fl.model import GenericCNN as Net
from medical_fl.utils import set_parameters
from medical_fl.data import load_data_iid, load_data_niid
from flwr.common import NDArrays, Scalar


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            print(len(images))
            images, labels = images.cuda().float(), labels.cuda().long()
            outputs = net(images)
            labels = labels.squeeze()
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


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


def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().cuda()
    _, _, testloader = load_data_iid(0, 1)
    set_parameters(net, parameters) # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}
