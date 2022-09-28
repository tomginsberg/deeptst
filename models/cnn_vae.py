import torch
import torchvision
from torch.autograd import Variable
from torch import nn
import pytorch_lightning as pl
import wandb
import torch.nn.functional as F

"""
This code is adapted from https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/model.py
"""


class VAE(pl.LightningModule):
    def __init__(self, image_size=32, channel_num=3, kernel_num=128, z_size=128,
                 lr=3e-04, weight_decay=1e-5, reconstruction_loss_weight=15,
                 reconstruction_loss='bce'):
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

    # def training_epoch_end(self, outputs) -> None:
    #     im_data = self.sample(16)
    #     grid = torchvision.utils.make_grid(im_data, nrow=4, normalize=True)
    #     self.logger.experiment.log({'samples': [wandb.Image(grid)]})

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


class ConvVAE(pl.LightningModule):
    def __init__(self, h_dim=512, z_dim=512, lr=1e-3, weight_decay=1e-5, reconstruction_loss_weight=15):
        super(ConvVAE, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.save_hyperparameters()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, h_dim)
        self.fc_bn1 = nn.BatchNorm1d(h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc_bn3 = nn.BatchNorm1d(h_dim)
        self.fc4 = nn.Linear(h_dim, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bce = nn.BCELoss(reduction='mean')

    def _encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mean = self.fc21(fc1)
        logvar = self.fc22(fc1)
        return mean, logvar

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        conv8 = self.sigmoid(self.conv8(conv7))
        return conv8.view(-1, 3, 32, 32)

    @staticmethod
    def _reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        mean, logvar = self._encode(x)
        z = self._reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def reconstruction_loss(self, x_reconstructed, x):
        return self.bce(x_reconstructed, x)

    @staticmethod
    def kl_divergence_loss(mean, logvar):
        return ((mean ** 2 + logvar.exp() - 1 - logvar) / 2).mean()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, logvar = self(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        kl_divergence = self.kl_divergence_loss(mean, logvar)
        loss = reconstruction_loss + self.reconstruction_loss_weight * kl_divergence
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, logvar = self(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        kl_divergence = self.kl_divergence_loss(mean, logvar)
        loss = reconstruction_loss + self.reconstruction_loss_weight * kl_divergence
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_epoch_end(self, outputs) -> None:
        samples = self.sample(16)
        grid = torchvision.utils.make_grid(samples, normalize=True, nrows=4)
        self.logger.experiment.log({'samples': [wandb.Image(grid)]}, commit=False)

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.hparams.z_dim)
        return self.decode(z)
