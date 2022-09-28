import argparse
import os

from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print('Evaluating...')

print('Evaluating...')
# from data import sample_data
print('Evaluating...')

from os.path import join

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.config import Config
from data.core import split_dataset

# from data import sample_data
print('Evaluating...')

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


def cifar10(split='train', normalize=True):
    if split not in {'train', 'val', 'test', 'all'}:
        raise ValueError(f'Invalid split: {split}')

    cfg = Config()

    train, rest = [torchvision.datasets.CIFAR10(
        root=cfg.get_dataset_path('cifar10'),
        train=x,
        download=False,
        transform=Compose([ToTensor(), Normalize(*get_mean_std(normalize))])
    ) for x in [True, False]]

    if split != 'train':
        val, test = split_dataset(rest, num_samples=int(len(rest) * 9 / 10), random_seed=0)
        if split == 'val':
            return val
        elif split == 'test':
            return test
        return train, val, test
    return train


def cifar10_1(normalize=True):
    cfg = Config()
    root = cfg.get_dataset_path('cifar10_1')
    data = torch.from_numpy(np.load(join(root, 'cifar10.1_v6_data.npy'))) / 255
    data = data.permute(0, 3, 1, 2)
    if normalize:
        data = data.sub(torch.tensor(MEAN).view(3, 1, 1)).div(torch.tensor(STD).view(3, 1, 1))
    labels = np.load(join(root, 'cifar10.1_v6_labels.npy'))
    return TensorDataset(data, torch.from_numpy(labels).long())


parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=10)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)

batch_size = args.n
weight_decay = 1e-5
learning_rate = 1e-3
lr_step = 1000
do_perm = True
epochs = 100
max_data_size = args.n
log_dir = args.log_dir

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

batch_size = min(max_data_size, batch_size)

img_size = 32
flattened_size = img_size * img_size * 3

device = torch.device("cuda:0")

fname = join(log_dir, 'cifar_result_' + str(max_data_size) + '.txt')


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    @staticmethod
    def sampling(mu, log_var):
        # std = torch.exp(0.5 * log_var)
        std = torch.sigmoid(log_var) * 2
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, flattened_size))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, torch.sigmoid(x.view(-1, flattened_size)), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(model, optimizer, train_loader, iteration):
    model.train()
    train_loss = []
    failed = False
    for ep in tqdm(range(epochs)):
        train_loss.append(0)
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            data = data.view((-1, flattened_size))
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)
            try:
                loss = loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                optimizer.step()
            except KeyboardInterrupt:
                exit()
            except RuntimeError:
                print('Invalid value encountered on loss function')
                failed = True
                break

            train_loss[-1] += loss.item()

    try:
        if failed:
            loss = train_loss[-2]
        else:
            loss = train_loss[-1]

        return loss / (len(train_loader.dataset))
    except IndexError:
        return False


def in_top_k(scores, alpha=0.05):
    k = int(len(scores) * alpha)
    indices = np.argsort(scores)[::-1]
    pos = np.where(indices == 0)[0][0]
    return pos <= k


def evaluate(samples=max_data_size, seed=0):
    path = join(args.log_dir, f'{samples=}_{seed=}.npy')
    path = os.path.realpath(path)
    print(f'Saving to path {path}')
    scores = []

    real_data = cifar10(split='test', normalize=True)
    cifar101, _ = split_dataset(cifar10_1(), num_samples=samples, random_seed=seed)

    for rep in range(100):

        indices = torch.randperm(len(real_data))[:samples]
        data1 = torch.stack([real_data[i][0] for i in indices], dim=0)

        data2 = torch.stack([cifar101[i][0] for i in range(samples)], dim=0)
        data2 = data2.type(torch.float32)

        if rep != 0:
            data_all = torch.cat([data1, data2], axis=0)

            data_all = data_all[np.random.permutation(range(data_all.shape[0]))]
            data1 = data_all[:samples]
            data2 = data_all[samples:]

        # train_data1 = torch.utils.data.TensorDataset(data1)
        train_loader1 = torch.utils.data.DataLoader(data1, batch_size=batch_size, shuffle=True)

        # train_data2 = torch.utils.data.TensorDataset(data2)
        train_loader2 = torch.utils.data.DataLoader(data2, batch_size=batch_size, shuffle=True)

        train_datam = torch.utils.data.TensorDataset(
            torch.cat([data1[:samples // 2], data2[:samples // 2]], axis=0))
        train_loaderm = torch.utils.data.DataLoader(train_datam, batch_size=batch_size, shuffle=True)

        model = VAE(x_dim=flattened_size, h_dim1=128, h_dim2=64, z_dim=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss2 = train(model=model, optimizer=optimizer, train_loader=train_loader2, iteration=rep)
        if loss2 is False:
            continue

        model = VAE(x_dim=flattened_size, h_dim1=128, h_dim2=64, z_dim=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lossm = train(model=model, optimizer=optimizer, train_loader=train_loaderm, iteration=rep)
        if lossm is False:
            continue

        model = VAE(x_dim=flattened_size, h_dim1=128, h_dim2=64, z_dim=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss1 = train(model=model, optimizer=optimizer, train_loader=train_loader1, iteration=rep)
        if loss1 is False:
            continue

        vdiv = np.mean(lossm) - min(np.mean(loss1), np.mean(loss2))

        print('Iteration - ', rep, ' vdiv - ', vdiv)
        scores.append(vdiv)
        np.save(path, np.array(scores))

    return in_top_k(scores, alpha=0.05)


def get_scores():
    print('total runs - ', args.runs)
    test_power_val = []
    for evaluation in range(args.runs):
        test_power_val.append(evaluate(seed=evaluation, samples=args.n) * 1.0)


if __name__ == "__main__":
    get_scores()
