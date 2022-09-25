import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.sample_data import cifar10
from models.cnn_vae import VAE

data = cifar10(split='train', normalize=False, augment=True)
dl = DataLoader(data, shuffle=True, batch_size=32, num_workers=32)

pl.seed_everything(0)
wandb_logger = WandbLogger(project="hdiv")
trainer = Trainer(logger=wandb_logger,
                  gpus=[1],
                  auto_select_gpus=True,
                  max_epochs=10,
                  log_every_n_steps=5,
                  callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints/cnn_vae/cifar',
                                                          monitor='train_loss',
                                                          mode='min',
                                                          save_last=True)])
model = VAE()
trainer.fit(model, dl)
