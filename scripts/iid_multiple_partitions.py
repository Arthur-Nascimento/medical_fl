import warnings

from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from medical_fl import train, test, evaluate
from medical_fl.data import load_data_iid, load_data_niid
from medical_fl.utils import get_parameters, set_parameters
from medical_fl.model import GenericCNN as Net
from medical_fl.client import FlowerClient
from medical_fl.strategy import CustomFedAvg
from medical_fl.transforms import ToTensor
from medical_fl.view import plot_distribution, print_samples_per_client
from medical_fl.utils import write_history

from pathlib import Path

# Importar o MedMNIST
from medmnist import OrganMNIST3D


# Desativar um aviso comum do Matplotlib no MedMNIST
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32

filename = "history.csv"
accuracies_csv = Path.cwd().parent / filename
accuracies_csv.touch()
with accuracies_csv.open(mode='w') as f:
    f.write("")


divisions = [5, 6, 7]
for NUM_PARTITIONS in divisions:
    # Função para criar clientes (client_fn)
    def client_fn(context: Context) -> Client:
        """Cria um Flower client para um dado client ID."""
        net = Net().to(DEVICE)
        # Cada cliente recebe seu próprio DataLoader de treino
        partition_id = context.node_config['partition-id']
        num_partitions = context.node_config['num-partitions']
        train_loader, val_loader, _ = load_data_iid(partition_id=partition_id, num_partitions=num_partitions, transforms=ToTensor())

        return FlowerClient(partition_id, net, train_loader, val_loader).to_client()


    def weighted_fit_average(metrics):
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        accuracy = sum(accuracies) / sum(examples)
        loss = sum(losses) / sum(examples)
        write_history("fit", accuracy, loss, filename='history.csv')

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": accuracy}


    def weighted_eval_average(metrics):
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        accuracy = sum(accuracies) / sum(examples)
        loss = sum(losses) / sum(examples)
        write_history("eval", accuracy, loss, filename="history.csv")

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": accuracy}


    def fit_config(server_round: int):
        """Return training configuration dict for each round

        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,
            "local_epochs": 3,
            }
        return config


    def eval_config(server_round: int):
        config = {
            "server_round": server_round,
            }
        return config


    params = get_parameters(Net())


    def server_fn(context: Context) -> ServerAppComponents:
        # min_available = context.run_node('min-available-clients')
        # num_partitions = context.node_config['num-partitions']
        strategy = CustomFedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            initial_parameters=ndarrays_to_parameters(params),
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            fit_metrics_aggregation_fn=weighted_fit_average,
            evaluate_metrics_aggregation_fn=weighted_eval_average,
        )
        config = ServerConfig(num_rounds=10)
        return ServerAppComponents(strategy=strategy, config=config)


    server = ServerApp(server_fn=server_fn)


    client = ClientApp(client_fn=client_fn)


    # NUM_PARTITIONS = 5
    NUM_CLIENTS = NUM_PARTITIONS
    print_samples_per_client(NUM_PARTITIONS=NUM_PARTITIONS, iid=True)
    plot_distribution(NUM_PARTITIONS=NUM_PARTITIONS, iid=True)

    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1}}

    # Iniciar a simulação
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
