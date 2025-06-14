import torchvision.transforms as transforms
from torch.data.utils import DataLoader


def load_datasets(partition_id: int, NUM_CLIENTS: int, BATCH_SIZE: int):
    fds = FederatedDataset(dataset='cifar10', partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # We will use this functions to dataset.with_transform(apply_tranforms)
        # The transforms object is exactly the same
        print(batch.keys())
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val  for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader
