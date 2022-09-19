from os.path import join

import os

import numpy as np
import torch
from scipy.io import loadmat

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


def uci_heart_xgb(data=None):
    if data is None:
        cfg = Config()
        data = loadmat(join(cfg.get_dataset_path('uci_heart'), 'uci_heart_processed.mat'))

    data_dict = {}
    for key in ['train', 'iid_test', 'val', 'ood_test']:
        data_dict[key] = xgb.DMatrix(data[f'{key}_data'], label=data[f'{key}_labels'])
    return data_dict


def uci_heart_numpy():
    cfg = Config()
    return loadmat(join(cfg.get_dataset_path('uci_heart'), 'uci_heart_processed.mat'))


from models.pretrained import xgb_trained_on_uci_heart
import pandas as pd

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist'
}

DATA_NUMPY = uci_heart_numpy()
DATA_XGB = uci_heart_xgb(data=DATA_NUMPY)
BASE_MODEL = xgb_trained_on_uci_heart(seed=0)


def train_and_test(data=DATA_XGB, params=DEFAULT_PARAMS, model_path='/voyager/datasets/UCI', n_seeds=10):
    # train <n_seeds> xgb models on the uci dataset and evaluate their in and out of distribution AUC
    iid_auc = []
    ood_auc = []

    for seed in range(n_seeds):
        path = os.path.join(model_path, f'uci_heart_{seed}.model')
        if os.path.exists(path):
            bst = xgb.Booster()
            bst.load_model(path)
            iid_auc.append(float(bst.eval(data['iid_test']).split(':')[1]))
            ood_auc.append(float(bst.eval(data['ood_test']).split(':')[1]))
        else:
            evallist = [(data['val'], 'eval'), (data['train'], 'train')]
            params['seed'] = seed
            num_round = 10
            bst = xgb.train(params, data['train'], num_round, evallist)

            iid_auc.append(float(bst.eval(data['iid_test']).split(':')[1]))
            ood_auc.append(float(bst.eval(data['ood_test']).split(':')[1]))

            bst.save_model(path)

    iid_auc = np.array(iid_auc)
    ood_auc = np.array(ood_auc)
    print(f'IID: {np.mean(iid_auc):.3f} \\pm {np.std(iid_auc):.3f}')
    print(f'OOD: {np.mean(ood_auc):.3f} \\pm {np.std(ood_auc):.3f}')
