from flwr.client import NumPyClient
from medical_fl.utils import get_parameters, set_parameters
from medical_fl import train
from medical_fl.eval import test


# 4. Definição do Cliente Flower
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
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
