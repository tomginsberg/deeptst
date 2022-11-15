from os.path import join
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from tqdm import tqdm

from utils.config import Config
from data.core import split_dataset
from models import pretrained

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_mean_std(normalize=True):
    if normalize:
        return MEAN, STD
    else:
        return (0, 0, 0), (1, 1, 1)


def download_cifar(root):
    cfg = Config()
    torchvision.datasets.CIFAR10(root=root, download=True)
    cfg.write_dataset_path('cifar10', root)


def cifar10(split='train', normalize=True, augment=False):
    if split not in ['train', 'val', 'test', 'all']:
        raise ValueError(f'Invalid split: {split}')

    cfg = Config()
    transform = []
    if augment:
        transform.append(RandomCrop(32, padding=4))
        transform.append(RandomHorizontalFlip())

    transform.append(ToTensor())
    if normalize:
        transform.append(Normalize(MEAN, STD))

    train, rest = [torchvision.datasets.CIFAR10(
        root=cfg.get_dataset_path('cifar10'),
        train=x,
        download=False,
        transform=Compose(transform)
    ) for x in [True, False]]

    if split != 'train':
        val, test = split_dataset(rest, num_samples=int(len(rest) * 9 / 10), random_seed=0)
        if split == 'val':
            return val
        elif split == 'test':
            return test
        return train, val, test
    return train


def cifar10_1(normalize=True, format_spec='list'):
    cfg = Config()
    root = cfg.get_dataset_path('cifar10_1')
    data = torch.from_numpy(np.load(join(root, 'cifar10.1_v6_data.npy'))) / 255
    data = data.permute(0, 3, 1, 2)
    if normalize:
        data = data.sub(torch.tensor(MEAN).view(3, 1, 1)).div(torch.tensor(STD).view(3, 1, 1))
    labels = np.load(join(root, 'cifar10.1_v6_labels.npy'))
    labels = torch.from_numpy(labels).long()
    if format_spec == 'tensor_dataset':
        return TensorDataset(data, labels)
    if format_spec == 'list':
        return [(d, l.item()) for d, l in zip(data, labels)]


def cifar10_features(split='train'):
    cfg = Config()
    root = cfg.get_dataset_path('cifar10_features')
    data = torch.load(join(root, f'{split}.pt'))
    labels = torch.load(join(root, f'{split}_labels.pt'))
    return data.numpy(), labels.numpy()


def featurize(root='/voyager/datasets/cifar10_features'):
    os.makedirs(root, exist_ok=True)
    device = torch.device('cuda:0')
    model = pretrained.resnet18_trained_on_cifar10().model
    model.eval()
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    model = model.to(device)
    for split in ['train', 'val', 'test', 'cifar10_1']:
        print(f'Featurizing {split}...')
        if split == 'cifar10_1':
            dataset = cifar10_1()
        else:
            dataset = cifar10(split=split)
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=16)
        features = []
        labels = []
        for batch in tqdm(loader):
            features.append(model(batch[0].to(device)).detach().cpu())
            labels.append(batch[1])
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        torch.save(features, join(root, f'{split}.pt'))
        torch.save(labels, join(root, f'{split}_labels.pt'))

    cfg = Config()
    cfg.write_dataset_path('cifar10_features', root)


if __name__ == '__main__':
    # validate performance on cifar10_1
    from models import pretrained
    from utils.inference import LabelCollector
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    model = pretrained.resnet18_trained_on_cifar10().model
    lc = LabelCollector(model)
    dl = DataLoader(cifar10_1(normalize=True), batch_size=2500, num_workers=16)
    pl.Trainer(gpus=1).validate(lc, dl, verbose=True)
    print(lc.compute_accuracy())
