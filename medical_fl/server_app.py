from medical_fl.task import get_parameters, set_parameters
from medical_fl.eval import evaluate
from medical_fl.strategy import CustomFedAvg
from medical_fl.model import GenericCNN as Net
from medical_fl.utils import write_history


from flwr.server import ServerAppComponents, ServerConfig, ServerApp
from flwr.common import Context, ndarrays_to_parameters


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 3,
        }
    return config


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": accuracy}


params = get_parameters(Net())


def server_fn(context: Context) -> ServerAppComponents:

    num_partitions = context.get_node('')
    strategy = CustomFedAvg(
        fraction_fit = 1,
        fraction_evaluate = 1,
        min_fit_clients = 3,
        min_evaluate_clients = 3,
        min_available_clients = 2,
        initial_parameters = ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=50)
    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)