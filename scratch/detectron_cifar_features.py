from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

from data.sample_data.cifar import cifar10_features
from models.pretrained import xgb_trained_on_cifar_features
from utils.detectron.modules import EarlyStopper

DEFAULT_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'merror',
    'num_class': 10,
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist',
}
BASE_MODEL = xgb_trained_on_cifar_features()


def all_but_n(n: int, num_classes: int) -> list[int]:
    return [i for i in range(num_classes) if i != n]


def invert_labels(labels: np.ndarray, num_classes=10) -> np.ndarray:
    return np.array([all_but_n(n, num_classes) for n in labels]).flatten()


def detectron_tst(train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
                  q: tuple[np.ndarray, np.ndarray], ensemble_size=10,
                  xgb_params=DEFAULT_PARAMS, base_model=BASE_MODEL, num_rounds=10,
                  patience=3, num_classes=10):
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
    q_pseudo_labels = q_pseudo_probabilities.argmax(1)
    inverted_q_pseudo_labels = invert_labels(q_pseudo_labels, num_classes=num_classes)
    duplicated_data = np.concatenate([np.array([x] * (num_classes - 1)) for x in q_data])
    q_len = len(duplicated_data)

    # create the weighted dataset for training the detectron
    pq_data = xgb.DMatrix(
        data=np.concatenate([train_data, duplicated_data]),
        label=np.concatenate([train_labels, inverted_q_pseudo_labels]),
        weight=np.concatenate(
            [np.ones_like(train_labels), 1 / (q_len + 1) * np.ones(q_len)]
        )
    )

    # set up the validation data
    val_dmatrix = xgb.DMatrix(val_data, val_labels)
    evallist = [(val_dmatrix, 'eval')]

    # evaluate the base model on test and auc data
    record.append({
        'ensemble_idx': 0,
        'val_acc': eval(base_model.eval(val_dmatrix).split(':')[1]),
        'test_acc': eval(base_model.eval(q_labeled).split(':')[1]),
        'rejection_rate': 0,
        'test_probabilities': q_pseudo_probabilities,
        'count': N
    })
    stopper = EarlyStopper(patience=patience, mode='min')
    stopper.update(N)

    # train the ensemble
    for i in range(1, ensemble_size + 1):
        # train the next model in the ensemble
        xgb_params.update({'seed': i})
        detector = xgb.train(xgb_params, pq_data, num_rounds, evals=evallist, verbose_eval=False)

        # evaluate the detector on the test data
        q_unlabeled = xgb.DMatrix(q_data)
        mask = ((detector.predict(q_unlabeled).argmax(1)) == q_pseudo_labels)

        # filter data to exclude the not rejected samples
        q_data = q_data[mask]
        q_pseudo_labels = q_pseudo_labels[mask]
        n = len(q_data)

        # log the results for this model
        record.append({'ensemble_idx': i,
                       'val_acc': 1 - float(detector.eval(val_dmatrix).split(':')[1]),
                       'test_acc': 1 - float(detector.eval(q_labeled).split(':')[1]),
                       'rejection_rate': 1 - n / N,
                       'test_probabilities': detector.predict(q_labeled),
                       'count': n})

        # break if no more data
        if n == 0:
            print(f'Converged to a rejection rate of 1 after {i} models')
            break

        if stopper.update(n):
            print(f'Early stopping: Converged after {i} models')
            break

        # update the training matrix
        inverted_q_pseudo_labels = invert_labels(q_pseudo_labels, num_classes=num_classes)
        duplicated_data = np.concatenate([np.array([x] * (num_classes - 1)) for x in q_data])
        q_len = len(duplicated_data)

        pq_data = xgb.DMatrix(
            data=np.concatenate([train_data, duplicated_data]),
            label=np.concatenate([train_labels, inverted_q_pseudo_labels]),
            weight=np.concatenate(
                [np.ones_like(train_labels), 1 / (q_len + 1) * np.ones(q_len)]
            )
        )

    return record


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--seeds', default=['0', '100'], nargs='+')
    parser.add_argument('--samples', default=[10, 20, 50], nargs='+')
    parser.add_argument('--splits', default=['p', 'q'], nargs='+')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    if os.path.exists(run_dir := os.path.join('results', args.run_name)) and not args.resume:
        raise ValueError(f'Run name <{args.run_name}> already exists')
    elif os.path.exists(run_dir) and args.resume:
        print(f'Resuming run <{args.run_name}>')
    else:
        os.makedirs(run_dir)
        print(f'Directory created for run: {run_dir}')

    seed_from, seed_to = int(args.seeds[0]), int(args.seeds[1])
    n_runs = len(args.samples) * len(args.splits) * (seed_to - seed_from)
    count = 0

    print(f'Staring {n_runs} runs')

    train = cifar10_features('train')
    val = cifar10_features('val')

    for N in map(int, args.samples):
        for dataset_name in args.splits:
            for seed in range(seed_from, seed_to):
                print(f'Starting run with {seed=}, {N=}, {dataset_name=}')
                if os.path.exists(f'results/{args.run_name}/test_{seed}_{dataset_name}_{N}.pt'):
                    print(f'Run already exists')
                    count += 1
                    continue

                # collect either p or q data and filter it to size N using random seed
                if dataset_name == 'p':
                    q = cifar10_features('test')
                else:
                    q = cifar10_features('cifar10_1')

                # randomly sample N elements from q
                idx = np.random.RandomState(seed).permutation(len(q[0]))[:N]
                q = q[0][idx, :], q[1][idx]
                res = detectron_tst(train=train, val=val, q=q)

                for r in res:
                    r.update({'seed': seed, 'N': N, 'dataset': dataset_name})
                    for k, v in r.items():
                        if isinstance(v, np.ndarray):
                            r[k] = torch.from_numpy(v)

                torch.save(res, f'results/{args.run_name}/test_{seed}_{dataset_name}_{N}.pt')

                count += 1
                print(f'Run {count}/{n_runs} complete, final rejection rate: {res[-1]["rejection_rate"]}')
