from typing import Any, Tuple

import torch

from .modules import DomainAdversarialNetwork, DANNLoader
from models import pretrained
from models.classifier import TorchvisionClassifier
from data.sample_data import cifar
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class DannCifar(pl.LightningModule):

    def __int__(self,
                domain_penalty=1,
                featurizer_lr=0.001,
                classifier_lr=0.01,
                discriminator_lr=0.01,
                batch_size=512,
                num_workers=16):
        featurizer = pretrained.resnet18_trained_on_cifar10().model
        classifier = torch.nn.Linear(featurizer.fc.in_features, 10)
        featurizer.fc = torch.nn.Identity()

        self.dann = DomainAdversarialNetwork(featurizer=featurizer, classifier=classifier, n_domains=2)
        self.domain_penalty = domain_penalty
        self.featurizer_lr = featurizer_lr
        self.classifier_lr = classifier_lr
        self.discriminator_lr = discriminator_lr
        self.id_class_acc = torchmetrics.Accuracy()
        self.od_class_acc = torchmetrics.Accuracy()
        self.domain_acc = torchmetrics.Accuracy()

        self.train_loader = DANNLoader(
            train=cifar.cifar10(split='train', normalize=True, augment=False),
            test=cifar.cifar10_1(normalize=True, format_spec='list'),
            val=cifar.cifar10(split='val', normalize=True, augment=False)
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dann(x)

    def training_step(self, batch, batch_idx):
        x, y, d = batch
        y_pred, domains_pred = self.dann(x)
        loss = F.cross_entropy(y_pred[d == 0], y[d == 0]) + F.cross_entropy(domains_pred, d) * self.domain_penalty
        self.log('train_loss', loss)
        self.id_class_acc(y_pred[d == 0], y[d == 0])
        self.od_class_acc(y_pred[d == 1], y[d == 1])
        self.domain_acc(domains_pred, d)
        return loss

    def training_epoch_end(self, outputs):
        self.log('id_class_acc', self.id_class_acc.compute())
        self.log('od_class_acc', self.od_class_acc.compute())
        self.log('domain_acc', self.domain_acc.compute())
        self.id_class_acc.reset()
        self.od_class_acc.reset()
        self.domain_acc.reset()

    def configure_optimizers(self):
        params = self.dann.get_parameters_with_lr(featurizer_lr=self.featurizer_lr, classifier_lr=self.classifier_lr,
                                                  discriminator_lr=self.discriminator_lr)
        return [torch.optim.Adam(*x) for x in params]

    def train_dataloader(self):
        cifar.cifar10(split='train', normalize=True)
