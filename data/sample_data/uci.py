from os.path import join

import torch
from utils.config import Config
from torch.utils.data import TensorDataset

import xgboost as xgb


def uci_heart(split='train', dataset_format='torch'):
    # wget https://github.com/tomginsberg/detectron/blob/main/data/uci_heart_torch.pt
    if split not in {'train', 'val', 'iid_test', 'ood_test'}:
        raise ValueError(f'Invalid split: {split}')

    cfg = Config()
    data = torch.load(join(cfg.get_dataset_path('uci_heart'), 'uci_heart_torch.pt'))
    data = data[split]
    data, labels = list(zip(*data))
    data = torch.stack(data)
    labels = torch.tensor(labels)

    if dataset_format == 'torch':
        return TensorDataset(data, labels)

    elif dataset_format in {'xgboost', 'xbg'}:
        return xgb.DMatrix(data.numpy(), label=labels.numpy())

    elif dataset_format == 'numpy':
        return data.numpy(), labels.numpy()

    else:
        raise ValueError(f'Unknown dataset format: {dataset_format}')
