import torch


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).float()
