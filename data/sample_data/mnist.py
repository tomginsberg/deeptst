from os.path import join

import torch
import torchvision
from utils.config import Config

import pickle

from torch.utils.data import TensorDataset


def mnist(train=True):
    cfg = Config()
    return torchvision.datasets.MNIST(
        root=cfg.get_dataset_path('mnist'),
        train=train,
        download=False,
        transform=torchvision.transforms.ToTensor()
    )


def fake_mnist():
    # gdown https://drive.google.com/uc?id=13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5
    cfg = Config()
    file = join(cfg.get_dataset_path('fake_mnist'), 'Fake_MNIST_data_EP100_N10000.pckl')
    data = (pickle.load(open(file, 'rb'))[0] + 1) / 2
    data = torch.from_numpy(data).float()
    return TensorDataset(data)
