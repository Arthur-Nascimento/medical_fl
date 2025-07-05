import matplotlib.pyplot as plt
import numpy as np

from medmnist import OrganMNIST3D
from medical_fl.data import load_data_iid, load_data_niid
from medical_fl.transforms import ToTensor


def plot_distribution(NUM_PARTITIONS = 5, alpha = 0.2, iid = True, tipo='train'):
    num_sample_partition = {i: [] for i in range(NUM_PARTITIONS)}
    dataset = OrganMNIST3D(split=tipo)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(NUM_PARTITIONS):
        if iid:
            trainloader, valloader, _ = load_data_iid(partition_id=i, num_partitions=NUM_PARTITIONS, transforms=ToTensor())
        else:
            trainloader, valloader, _ = load_data_niid(partition_id=i, num_partitions=NUM_PARTITIONS, transforms=ToTensor(), alpha=alpha)
        if tipo == 'train':
            num_sample_partition[i].extend(trainloader.dataset[:][1].squeeze())
        else:
            num_sample_partition[i].extend(valloader.dataset[:][1].squeeze())
    for i in range(NUM_PARTITIONS):
        label_counts = np.bincount(num_sample_partition[i], minlength=11)
        plt.bar(np.arange(11) + i * 0.1, label_counts, width=0.1, label=f'Cliente {i}')
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(dataset.info['label'].items(), rotation=45, ha="right")
    ax.set_ylabel("Número de Amostras")
    if iid:
        ax.set_title(f"Distribuição de Rótulos por Cliente: IID ({tipo})")
    else:    
        ax.set_title(f"Distribuição de Rótulos por Cliente: Não-IID (alpha={alpha}) ({tipo})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_samples_per_client(NUM_PARTITIONS = 3, iid = False, alpha=0.2):
    for i in range(NUM_PARTITIONS):
        trainloader, valloader, testloader = load_data_iid(i, NUM_PARTITIONS, transforms=ToTensor()) if iid else load_data_niid(i, NUM_PARTITIONS, transforms=ToTensor(), alpha=alpha)
        print(f"Train samples: {sum([len(trainloader.dataset)])}")
        print(f"Val samples: {sum([len(valloader.dataset)])}")
        print(f"Test samples: {len(testloader.dataset)}")
        print(f"Distribuição: {np.unique(trainloader.dataset[:][1], return_counts=True)}")
        print("____________________________________________________________")
