from data import sample_data
import torch

import pytorch_lightning as pl
from models import pretrained
import torchmetrics
from data.core import split_dataset
from torch.utils.data import DataLoader, Dataset
from scipy.stats import binom_test
import numpy as np


class DomainClassifier(pl.LightningModule):
    def __init__(self, dataset, lr=1e-3):
        super().__init__()
        if dataset == 'camelyon':
            self.model = pretrained.camelyon_model(domain_classifier=True).model
        elif dataset == 'cifar':
            self.model = pretrained.resnet18_trained_on_cifar10(domain_classifier=True).model
        else:
            raise ValueError(f'Invalid dataset {dataset}')

        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_acc', self.accuracy(y_hat, y))
        return loss

    def reset(self):
        self.accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DomainClassifierDataset(Dataset):
    def __init__(self, p, q):
        self.p = [x for x in p]  # prefetch data
        self.q = [x for x in q]
        self.p_labels = torch.zeros(len(p)).long()
        self.q_labels = torch.ones(len(q)).long()
        self.x = self.p + self.q
        self.y = torch.cat([self.p_labels, self.q_labels])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx][0], self.y[idx]


if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--dataset', type=str, default='cifar')
    args = parser.parse_args()

    os.makedirs(f'results/{args.run_name}', exist_ok=True)
    if args.dataset == 'cifar':
        p_all = sample_data.cifar10(split='test')
        q_all = sample_data.cifar10_1()
    elif args.dataset == 'camelyon':
        p_all = sample_data.camelyon(split='test', quantized=True)
        q_all = sample_data.camelyon(split='harmful', quantized=True)
    else:
        raise ValueError(f'Invalid dataset {args.dataset}')

    N = args.n

    for seed in range(100):
        data = []
        p, _ = split_dataset(p_all, N, seed)
        p1, p2 = split_dataset(p, N // 2, seed)
        q, _ = split_dataset(q_all, N, seed)
        q1, q2 = split_dataset(q, N // 2, seed)

        d1 = DomainClassifierDataset(p1, q1)
        d2 = DomainClassifierDataset(p2, q2)
        model = DomainClassifier(dataset=args.dataset, lr=1e-3)
        result = []
        tr = pl.Trainer(max_epochs=10, gpus=[0],
                        enable_checkpointing=False,
                        enable_model_summary=False,
                        logger=False)
        acc = tr.test(model, DataLoader(d2, batch_size=N, shuffle=False, num_workers=4), verbose=False)[0][
            'test_acc']
        data.append([0, acc])

        for train_round in range(1, 41):
            tr = pl.Trainer(max_epochs=10, gpus=[0],
                            enable_checkpointing=False,
                            enable_model_summary=False,
                            logger=False)

            tr.fit(model, DataLoader(d1, batch_size=N, shuffle=True, num_workers=4))
            acc = tr.test(model, DataLoader(d2, batch_size=N, shuffle=False, num_workers=4), verbose=False)[0][
                'test_acc']
            data.append([train_round, acc])
            print(data)
            np.save(f'results/{args.run_name}/{N=}_{seed=}.npy', np.array(data))
