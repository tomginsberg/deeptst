import os.path

from utils.config import Config
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def download_camelyon(root_dir):
    cfg = Config()
    Camelyon17Dataset(root_dir=root_dir, download=True)
    cfg.write_dataset_path('camelyon17', root_dir)


def camelyon(split='train'):
    cfg = Config()
    try:
        dataset = Camelyon17Dataset(root_dir=cfg.get_dataset_path('camelyon17'), download=False)
    except FileNotFoundError:
        response = input('Camelyon17 dataset not found. Download? [y/n]: ')
        if response == 'y':
            path = input(f'Enter root directory for download or press '
                         f'Enter to use default [{cfg.get_dataset_path()}]: ')
            if path == '':
                path = cfg.get_dataset_path()
            if not os.path.exists(p := os.path.realpath(path)):
                raise NotADirectoryError(p)

            download_camelyon(path)
            dataset = Camelyon17Dataset(root_dir=cfg.get_dataset_path('camelyon17'), download=False)
        else:
            print('Download Camelyon17 manually and update `camelyon17` in config.json')
            raise FileNotFoundError('Camelyon17 dataset not found.')
    return dataset.get_subset(split, transform=Compose([Resize((224, 224)), ToTensor()]))
