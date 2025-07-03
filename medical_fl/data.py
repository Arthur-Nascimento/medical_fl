import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from medmnist import OrganMNIST3D


def load_data_iid(partition_id: int,
                  num_partitions: int,
                  transforms=None,
                  ):
    # Transformações para o dataset
    data_transform = transforms

    # Carregar o dataset de treino completo do MedMNIST
    full_train_dataset = OrganMNIST3D(split="train", transform=data_transform, download=True)
    full_val_dataset = OrganMNIST3D(split="val", transform=data_transform, download=True)

    def partition_data(data):
        num_images = len(data)
        partition_size = num_images // num_partitions
        lengths = [partition_size] * num_partitions
        # Caso divisão não exata
        remainder = num_images % num_partitions
        for i in range(remainder):
            lengths[i] += 1
        return lengths

    generator = torch.Generator().manual_seed(42)
    train_lengths = partition_data(full_train_dataset)
    val_lengths = partition_data(full_val_dataset)

    # Usar a função do PyTorch para dividir o dataset em partições não sobrepostas
    # Esta é a alternativa ao Partitioner do Flower
    # random_split não deve gerar amostras únicas para um mesmo cliente, necessário verificar
    train_partitions = random_split(full_train_dataset, train_lengths, generator=generator)
    val_partitions = random_split(full_val_dataset, val_lengths, generator=generator)

    # Criar um DataLoader para cada partição
    train_loaders = [DataLoader(part, batch_size=32, shuffle=True) for part in train_partitions]
    val_loaders = [DataLoader(part, batch_size=32, shuffle=True) for part in val_partitions]

    # Carregar o dataset de teste (geralmente é centralizado e não particionado)
    test_loader = DataLoader(OrganMNIST3D(split="test", download=True), batch_size=32)

    return train_loaders[partition_id], val_loaders[partition_id], test_loader


def load_data_niid(partition_id: int,
                   num_partitions: int,
                   transforms=None,
                   alpha=0.2,
                   seed=64,
                   ):
    """
    Particiona um dataset entre múltiplos clientes usando uma Distribuição de Dirichlet
    para simular um cenário Não-IID com Label Distribution Skew.

    Args:
        dataset: O dataset a ser particionado (ex: torchvision.datasets.CIFAR10).
        num_clients (int): O número de clientes para dividir os dados.
        alpha (float): O parâmetro de concentração da Dirichlet. Alpha pequeno = mais desbalanceado.

    Returns:
        dict: Um dicionário onde a chave é o ID do cliente e o valor é uma lista de
              índices de dados pertencentes àquele cliente.
    """
    # Transformações para o dataset
    data_dist_dict = {
        'cardio': [6, 9],
        'pneumo': [7, 8],
        'gastro': [0, 10],
        'urolo': [1, 2, 5],
        'bone': [3, 4],
    }
    data_transform = transforms

    # Carregar o dataset de treino completo do MedMNIST
    full_train_dataset = OrganMNIST3D(split="train", transform=data_transform, download=True)
    full_val_dataset = OrganMNIST3D(split="val", transform=data_transform, download=True)

    def split_idx(data):
        # Obter os rótulos de todo o dataset
        labels = data.labels
        num_classes = len(data.info['label'])
        client_partitions = {i: [] for i in range(num_partitions)}
        sampler = np.random.default_rng(seed=seed)
        distribution = sampler.dirichlet(np.repeat(alpha, num_partitions), num_classes)

        for i in range(num_classes):
            class_indices = np.where(labels == i)[0]
            proportion = (distribution[i]*len(class_indices)).astype(int)
            missing = len(class_indices) - np.sum(proportion)
            proportion[np.argmax(proportion)] += missing
            assert np.sum(proportion) == len(class_indices)

            position = 0
            for client_id in range(num_partitions):
                client_samples = proportion[client_id]
                client_partitions[client_id].extend(class_indices[position:position+client_samples])
        return client_partitions

    train_partitions = split_idx(full_train_dataset)
    train_set = torch.utils.data.Subset(full_train_dataset, train_partitions[partition_id])
    trainloader = DataLoader(train_set, batch_size=32)
    val_partitions = split_idx(full_val_dataset)
    val_set = torch.utils.data.Subset(full_train_dataset, val_partitions[partition_id])
    valloader = DataLoader(val_set, batch_size=32)

    testloader = DataLoader(OrganMNIST3D(split="test", download=True), batch_size=32)
    return trainloader, valloader, testloader