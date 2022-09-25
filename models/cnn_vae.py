import torch
import torchvision
from torch.autograd import Variable
from torch import nn
import pytorch_lightning as pl
import wandb

"""
This code is adapted from https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/model.py
"""


class VAE(pl.LightningModule):
    def __init__(self, image_size=32, channel_num=3, kernel_num=128, z_size=128,
                 lr=3e-04, weight_decay=1e-5, reconstruction_loss_weight=15,
                 reconstruction_loss='mse'):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.reconstruction_loss_weight = reconstruction_loss_weight
        if reconstruction_loss == 'mse':
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss == 'bce':
            self.reconstruction_loss_fn = nn.BCELoss()
        else:
            raise ValueError('invalid reconstruction loss')
        self.save_hyperparameters()

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return self.reconstruction_loss_weight * self.reconstruction_loss_fn(x_reconstructed, x)

    @staticmethod
    def kl_divergence_loss(mean, logvar):
        return ((mean ** 2 + logvar.exp() - 1 - logvar) / 2).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self.forward(x)
        loss = self.reconstruction_loss(x_reconstructed, x) + self.kl_divergence_loss(mean, logvar)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        im_data = self.sample(16)
        grid = torchvision.utils.make_grid(im_data, nrow=4, normalize=False)
        self.logger.experiment.log({'samples': [wandb.Image(grid)]})

    def test_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self.forward(x)
        loss = self.reconstruction_loss(x_reconstructed, x) + self.kl_divergence_loss(mean, logvar)
        self.log('test_loss', loss)
        return loss

    # =====
    # Utils
    # =====

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
