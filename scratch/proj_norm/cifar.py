from os.path import join

import numpy as np
import pandas as pd
import torch.nn
import torchvision
from torch.utils.data import TensorDataset, random_split
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def split_dataset(dataset, num_samples: int, random_seed: int = 42):
    # noinspection PyTypeChecker
    return random_split(dataset, [num_samples, len(dataset) - num_samples],
                        generator=torch.Generator().manual_seed(random_seed))


def get_mean_std(normalize=True):
    if normalize:
        return MEAN, STD
    else:
        return (0, 0, 0), (1, 1, 1)


def cifar10(root='/voyager/datasets', split='train', normalize=True, augment=False):
    if split not in ['train', 'val', 'test', 'all']:
        raise ValueError(f'Invalid split: {split}')

    transform = []
    if augment:
        transform.append(RandomCrop(32, padding=4))
        transform.append(RandomHorizontalFlip())

    transform.append(ToTensor())
    if normalize:
        transform.append(Normalize(MEAN, STD))

    train, rest = [torchvision.datasets.CIFAR10(
        root=root,
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


def cifar10_1(root='/voyager/datasets/cifar-10-1', normalize=True, format_spec='list'):
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


def compute_proj_norm(model, dataset):
    """ Takes a model and a dataset and computes the projection norm of the model on the dataset """
    return 0


if __name__ == '__main__':

    # parameters for the experiment
    samples = [10, 20, 50]
    seeds = 100
    alpha = 0.05
    checkpoint_path = '/voyager/projects/tomginsberg/checkpoints/resnet18_cifar.pt'

    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    print(model.load_state_dict(torch.load(checkpoint_path)))
    model.eval()

    p_train, p_val, p_test_all = cifar10(split='all', normalize=True)  # loads the cifar 10 dataset
    # above model was trained on p_train + p_val

    q_all = cifar10_1(normalize=True)  # loads the cifar 10_1 dataset

    test_sets = {'p': p_test_all, 'q': q_all}

    proj_norms = {x: {'p': [], 'q': []} for x in samples}
    for N in samples:
        for seed in range(seeds):
            for test in ['p', 'q']:
                q, _ = split_dataset(test_sets[test], random_seed=seed, num_samples=N)
                proj_norm_val = compute_proj_norm(model, q)
                print(f'N={N}, seed={seed}, test={test}, proj_norm={proj_norm_val}')
                proj_norms[N][test].append(proj_norm_val)

    df = pd.DataFrame(proj_norms)
    df.to_json('proj_norms_cifar.json')

    # find quantile on in dist data
    print('-' * 60)
    for N in samples:
        thresh = np.quantile(proj_norms[N]['p'], alpha)
        power = (np.array(proj_norms[N]['q']) < thresh)
        print(f'N={N}, power={power.mean()} +- {power.std() / np.sqrt(seeds)}')
