from typing import List, OrderedDict
import numpy as np
import torch
from pathlib import Path
import pandas as pd


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def write_history(*args, filename):
    string = ""
    for arg in args:
        string += str(arg)
        string += ", "
    string += '\n'
    file = Path.cwd().parent / filename
    with file.open(mode="a") as f:
        f.write(string)


def _write_history(train_stage, node, loss, accuracy, round):
    string = f"{round}, {node}, {train_stage}, {loss}, {accuracy}\n"
    file = Path.cwd().parent / 'history.csv'
    with file.open(mode="a") as f:
        f.write(string)


def load_history_as_dict():
    header = ['round', 'id', 'stage', 'loss', 'accuracy']
    file = Path.cwd().parent / 'history.csv'
    df = pd.read_csv(file, sep=',')
