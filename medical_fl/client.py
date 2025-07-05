from medical_fl.utils import get_parameters, set_parameters
from medical_fl import train
from medical_fl.eval import test
from medical_fl.data import load_data_iid
from medical_fl.model import GenericCNN as Net


import torch

from flwr.client import NumPyClient, ClientApp, Client
from flwr.common import Context


from pathlib import Path

history_path = Path.cwd().parent / "history.csv"


# Definição do Cliente Flower
class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, testloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Reading from config
        server_round = config['server_round']
        local_epochs = config['local_epochs']

        # Use the values
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        loss, accuracy = test(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader.dataset), {"accuracy": float(accuracy), 'loss': loss}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        round = config['server_round']
        set_parameters(self.net, parameters)
        for batch in self.testloader:
            assert len(batch[1]) > 0
        loss, accuracy = test(self.net, self.testloader)
        # write_history("eval", self.pid, loss, accuracy, round)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy), "loss": loss}
