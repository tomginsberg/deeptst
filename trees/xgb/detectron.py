from __future__ import annotations

import numpy as np
import xgboost as xgb

from trees.defaults import XGB_PARAMS
from trees.utils import XGBDetectronDataModule, XGBDetectronRecord
from utils.detectron.modules import EarlyStopper
from tqdm import tqdm
from scipy.interpolate import interp1d


def detectron_test_statistics(
        train: tuple[np.ndarray, np.ndarray],
        val: tuple[np.ndarray, np.ndarray],
        q: tuple[np.ndarray, np.ndarray],
        base_model: xgb.Booster,
        sample_size: int,
        xgb_params=XGB_PARAMS,
        ensemble_size=10,
        calibration_runs=100,
        num_boost_round=10,
        patience=3,
        balance_train_classes=True
):
    """
    Run the Detectron algorithm for `seeds` times, and return
    :param train:
    :param val:
    :param q:
    :param base_model:
    :param sample_size:
    :param xgb_params: (trees.defaults.XGB_PARAMS) parameters for the XGBoost model
    :param ensemble_size: (10) the number of models in the ensemble
    :param calibration_runs: (100) the number of different random runs to perform,
        each run operates on a random sample from q
    :param num_boost_round: (10) xgb parameter for the number of boosting rounds
    :param patience: (3) number of ensemble rounds to wait without improvement in the rejection rate
    :param balance_train_classes: (True) If True, the training data will be automatically balanced using weights.
    :return: XGBDetectronRecord object containing all the information of this run
    """
    record = XGBDetectronRecord(sample_size)

    # gather the data
    train_data, train_labels = train
    val_data, val_labels = val
    q_data_all, q_labels_all = q

    # set up the validation data
    val_data = xgb.DMatrix(val_data, val_labels)

    for seed in tqdm(range(calibration_runs)):

        # randomly sample N elements from q
        idx = np.random.RandomState(seed).permutation(len(q[0]))[:sample_size]
        q_data, q_labels = q_data_all[idx, :], q_labels_all[idx]

        # store the test data
        N = len(q_data)
        q_labeled = xgb.DMatrix(q_data, label=q_labels)

        # evaluate the base model on the test data
        q_pseudo_probabilities = base_model.predict(q_labeled)
        q_pseudo_labels = q_pseudo_probabilities > 0.5

        # create the weighted dataset for training the detectron
        data_module = XGBDetectronDataModule(train_data=train_data,
                                             train_labels=train_labels,
                                             q_data=q_data,
                                             q_pseudo_labels=q_pseudo_labels,
                                             balance_train_classes=balance_train_classes)

        # evaluate the base model on test and auc data
        record.seed(seed)
        record.update(q_labeled=q_labeled, val_data=val_data, model=base_model, sample_size=N,
                      q_pseudo_probabilities=q_pseudo_probabilities)

        stopper = EarlyStopper(patience=patience, mode='min')
        stopper.update(N)

        # train the ensemble
        for i in range(1, ensemble_size + 1):
            # train the next model in the ensemble
            xgb_params.update({'seed': i})
            detector = xgb.train(xgb_params, data_module.dataset(),
                                 num_boost_round=num_boost_round,
                                 evals=[(val_data, 'eval')],
                                 verbose_eval=False)

            n = data_module.filter(detector)

            # log the results for this model
            record.update(q_labeled=q_labeled, val_data=val_data, model=detector, sample_size=n)

            # break if no more data
            if n == 0:
                print(f'Converged to a rejection rate of 1 after {i} models')
                break

            if stopper.update(n):
                print(f'Early stopping: Converged after {i} models')
                break

    record.freeze()
    return record


def detectron_dis_power(calibration_result: XGBDetectronRecord,
                        test_record: XGBDetectronRecord,
                        alpha=0.05,
                        max_ensemble_size=None):
    """
    Compute the discovery power of the detectron algorithm.
    :param calibration_result: (XGBDetectronRecord) the results of the calibration run
    :param test_record: (XGBDetectronRecord) the results of the test run
    :param alpha: (0.05) the significance level
    :param max_ensemble_size: (None) the maximum number of models in the ensemble to consider.
        If None, all models are considered.
    :return: the discovery power
    """
    cal_counts = calibration_result.counts(max_ensemble_size=max_ensemble_size)
    test_counts = test_record.counts(max_ensemble_size=max_ensemble_size)
    N = calibration_result.sample_size
    assert N == test_record.sample_size, 'The sample sizes of the calibration and test runs must be the same'

    fpr = (cal_counts >= np.arange(0, N + 2)[:, None]).mean(1)[::-1]
    tpr = (test_counts >= np.arange(0, N + 2)[:, None]).mean(1)[::-1]

    quantile = np.quantile(cal_counts, 1 - alpha)
    tpr_low = (test_counts > quantile).mean()
    tpr_high = (test_counts >= quantile).mean()

    fpr_low = (cal_counts > quantile).mean()
    fpr_high = (cal_counts >= quantile).mean()

    if fpr_high == fpr_low:
        tpr_at_alpha = tpr_high
    else:  # use linear interpolation if there is no threshold at alpha
        tpr_at_alpha = (tpr_high - tpr_low) / (fpr_high - fpr_low) * (alpha - fpr_low) + tpr_low

    return dict(tpr=tpr_at_alpha, auc=np.trapz(tpr, fpr))
