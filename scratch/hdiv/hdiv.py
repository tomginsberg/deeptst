import numpy as np
import pandas as pd
from data import sample_data

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.core import split_dataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from glob import glob
import random
from models.cnn_vae import VAE as CNNVAE


class VAE(pl.LightningModule):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, lr=1e-4, sigmoid_input=True):
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
        self.lr = lr
        self.sigmoid_input = sigmoid_input
        self.save_hyperparameters()

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        x_hat, mu, log_var = self(x)
        loss = loss_function(x_hat, x, mu, log_var)
        self.log('train_loss', loss)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.flatten(1)
        x_hat, mu, log_var = self(x)
        loss = loss_function(x_hat, x, mu, log_var, sigmoid_input=self.sigmoid_input)
        self.log('test_loss', loss)
        return {'loss': loss}


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var, sigmoid_input=True):
    if sigmoid_input:
        torch.sigmoid(x)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def pretrain_uci():
    data = sample_data.uci_heart(split='train')
    dl = DataLoader(data, shuffle=True, batch_size=64, num_workers=4)

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="hdiv")
    trainer = Trainer(logger=wandb_logger,
                      gpus=1,
                      auto_select_gpus=True,
                      max_epochs=300,
                      log_every_n_steps=1,
                      callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints/vae/uci',
                                                              monitor='train_loss',
                                                              mode='min')])
    model = VAE(x_dim=9, h_dim1=7, h_dim2=5, z_dim=3, lr=1e-4)
    trainer.fit(model, dl)


def pretrain_cifar():
    data = sample_data.cifar10(split='train', normalize=False)
    dl = DataLoader(data, shuffle=True, batch_size=256, num_workers=32)

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="hdiv")
    trainer = Trainer(logger=wandb_logger,
                      gpus=[1],
                      max_epochs=50,
                      log_every_n_steps=1,
                      callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints/vae/cifar',
                                                              monitor='train_loss',
                                                              mode='min')])
    model = VAE(x_dim=32 * 32 * 3, h_dim1=256, h_dim2=128, z_dim=64, lr=1e-3, sigmoid_input=False)
    trainer.fit(model, dl)


if __name__ == '__main__':
    import argparse
    import os

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int)
    argparser.add_argument('--run_name', type=str)
    argparser.add_argument('--ckpt', type=str)
    argparser.add_argument('--dataset', type=str, default='uci')
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--cnn', default=False, action='store_true')
    args = argparser.parse_args()

    N = args.n
    seeds = 100
    print(f'Starting {2 * seeds} runs for {N=} samples')

    if args.dataset == 'uci':
        p_data = sample_data.uci_heart(split='iid_test')
        q_data = sample_data.uci_heart(split='ood_test')
    elif args.dataset == 'cifar':
        p_data = sample_data.cifar10(split='test', normalize=False)
        q_data = sample_data.cifar10_1(normalize=False)
    elif args.dataset == 'camelyon':
        p_data = sample_data.camelyon(split='test')
        q_data = sample_data.camelyon(split='harmful')
    else:
        raise ValueError('Dataset not defined')

    os.makedirs('results/' + args.run_name, exist_ok=True)
    save_path = f'results/{args.run_name}/{args.dataset}_{N=}.npy'
    print(f'Results will be saved to {save_path}')


    def run(N, data, checkpoint=args.ckpt, gpu=args.gpu):
        dl = DataLoader(data, shuffle=True, batch_size=2 * N, num_workers=16)
        model = (CNNVAE if args.cnn else VAE).load_from_checkpoint(checkpoint)
        tr = pl.Trainer(gpus=[gpu],
                        max_epochs=50,
                        checkpoint_callback=None,
                        logger=False,
                        enable_model_summary=True,
                        enable_checkpointing=False)
        tr.fit(model, dl)
        return tr.test(model, dl, verbose=False)[0]['test_loss']


    p_losses = []
    q_losses = []
    pq_losses = []
    for seed in range(seeds):
        pl.seed_everything(seed)
        p, _ = split_dataset(p_data, N, seed)
        q, _ = split_dataset(q_data, N, seed)
        pq, _ = split_dataset(p + q, N, seed)

        p = [x for x in p]  # cache datasets
        q = [x for x in q]
        pq = [x for x in pq]

        p_loss = run(N, p)
        q_loss = run(N, q)
        pq_loss = run(N, pq)

        p_losses.append(p_loss)
        q_losses.append(q_loss)
        pq_losses.append(pq_loss)
        all_ = np.array([p_losses, q_losses, pq_losses])
        np.save(save_path, all_)
        print('p', p_loss, '\nq', q_loss, '\npq', pq_loss)

    p_stats = []
    q_stats = []
    for s in range(seeds):
        # sample 3 random elements from p_loss
        rnd = random.Random(s)
        p = rnd.sample(population=p_losses, k=3)
        p_stat = p[0] - min(p[1], p[2])
        p_stats.append(p_stat)

        q_stat = pq_losses[s] - min(p_losses[s], q_losses[s])
        q_stats.append(q_stat)

    tests = np.quantile(p_stats, 0.95) < q_stats
    res = np.array([tests.mean(), tests.std() / np.sqrt(seeds)])
    print(res)
    np.save(save_path.replace('.npy', '_res.npy'), res)
