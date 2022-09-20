from __future__ import annotations

import numpy as np
import xgboost as xgb

from data.sample_data.uci import uci_heart_xgb, uci_heart_numpy
from models.pretrained import xgb_trained_on_uci_heart

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


def entropy(x):
    epsilon = 1e-8
    e = (1 - x) * np.log(1 - x + epsilon) + x * np.log(x + epsilon)
    return -e


def ensemble_entropy(x):
    return entropy(np.mean(x, axis=0))


def detectron_tst(train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
                  q: tuple[np.ndarray, np.ndarray], ensemble_size=10,
                  xgb_params=DEFAULT_PARAMS, base_model=BASE_MODEL, num_rounds=10):
    record = []

    # gather the data
    train_data, train_labels = train
    val_data, val_labels = val
    q_data, q_labels = q

    # store the test data
    N = len(q_data)
    q_labeled = xgb.DMatrix(q_data, label=q_labels)

    # evaluate the base model on the test data
    q_pseudo_probabilities = base_model.predict(q_labeled)
    q_pseudo_labels = q_pseudo_probabilities > 0.5

    # create the weighted dataset for training the detectron
    pq_data = xgb.DMatrix(
        data=np.concatenate([train_data, q_data]),
        label=np.concatenate([train_labels, 1 - q_pseudo_labels]),
        weight=np.concatenate(
            [np.ones_like(train_labels), 1 / (N + 1) * np.ones(N)]
        )
    )

    # set up the validation data
    val_dmatrix = xgb.DMatrix(val_data, val_labels)
    evallist = [(val_dmatrix, 'eval')]

    # evaluate the base model on test and auc data
    record.append({
        'ensemble_idx': 0,
        'val_auc': eval(base_model.eval(val_dmatrix).split(':')[1]),
        'test_auc': eval(base_model.eval(q_labeled).split(':')[1]),
        'test_reject': 0,
        'test_probabilities': q_pseudo_probabilities
    })

    # train the ensemble
    for i in range(1, ensemble_size + 1):
        # train the next model in the ensemble
        xgb_params.update({'seed': i})
        detector = xgb.train(xgb_params, pq_data, num_rounds, evals=evallist, verbose_eval=False)

        # evaluate the detector on the test data
        q_unlabeled = xgb.DMatrix(q_data)
        mask = ((detector.predict(q_unlabeled) > 0.5) == q_pseudo_labels)

        # filter data to exclude the not rejected samples
        q_data = q_data[mask]
        q_pseudo_labels = q_pseudo_labels[mask]
        n = len(q_data)

        # log the results for this model
        record.append({'ensemble_idx': i,
                       'val_auc': eval(detector.eval(val_dmatrix).split(':')[1]),
                       'test_auc': eval(detector.eval(q_labeled).split(':')[1]),
                       'test_reject': 1 - n / N,
                       'test_probabilities': detector.predict(q_labeled)})

        # break if no more data
        if n == 0:
            print(f'Converged to a rejection rate of 1 after {i} models')
            break

        # update the training matrix
        pq_data = xgb.DMatrix(
            data=np.concatenate([train_data, q_data]),
            label=np.concatenate([train_labels, 1 - q_pseudo_labels]),
            weight=np.concatenate(
                [np.ones_like(train_labels), 1 / (n + 1) * np.ones(n)]
            )
        )

    return record


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--samples', default=[10, 20, 50], nargs='+')
    parser.add_argument('--splits', default=['p', 'q'], nargs='+')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    for N in map(int, args.samples):
        for dataset_name in args.splits:
            for seed in range(args.seeds):
                # collect either p or q data and filter it to size N using random seed
                if dataset_name == 'p':
                    data = DATA_XGB['iid_test_data']
                else:
                    data = DATA_XGB['ood_test_data']

                data = data.sample(n=N, random_state=seed)
