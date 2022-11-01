import xgboost as xgb
import numpy as np
import pandas as pd


class XGBDetectronDataModule:

    def __init__(self, train_data: np.ndarray, q_data, train_labels, q_pseudo_labels, balance_train_classes=True):
        """

        :param train_data:
        :param q_data:
        :param train_labels:
        :param q_pseudo_labels:
        :param balance_train_classes:
        :return:
        """
        self.N = len(q_data)

        self.train_data = train_data
        self.q_data = q_data
        self.train_labels = train_labels
        self.q_pseudo_labels = q_pseudo_labels

        if balance_train_classes:
            _, counts = np.unique(train_labels, return_counts=True)
            assert len(counts) == 2, 'Only binary classification is supported in v0.0.1'
            c_neg, c_pos = counts[0], counts[1]
            # make sure the average training weight is 1
            pos_weight, neg_weight = 2 * c_neg / (c_neg + c_pos), 2 * c_pos / (c_neg + c_pos)
            self.train_weights = np.array([pos_weight if label == 1 else neg_weight for label in train_labels])
        else:
            self.train_weights = np.ones_like(train_labels)

    def dataset(self):
        return xgb.DMatrix(
            data=np.concatenate([self.train_data, self.q_data]),
            label=np.concatenate([self.train_labels, 1 - self.q_pseudo_labels]),
            weight=np.concatenate(
                [self.train_weights, 1 / (self.N + 1) * np.ones(self.N)]
            )
        )

    def filter(self, detector):
        mask = ((detector.predict(xgb.DMatrix(self.q_data)) > 0.5) == self.q_pseudo_labels)

        # filter data to exclude the not rejected samples
        self.q_data = self.q_data[mask]
        self.q_pseudo_labels = self.q_pseudo_labels[mask]
        return len(self.q_data)


class XGBDetectronRecord:
    def __init__(self, sample_size):
        self.record = []
        self.sample_size = sample_size
        self.idx = 0
        self._seed = None

    def seed(self, seed):
        self._seed = seed
        self.idx = 0

    def update(self, q_labeled, val_data, sample_size, model,
               q_pseudo_probabilities=None):
        assert self._seed is not None, 'Seed must be set before updating the record'
        self.record.append({
            'ensemble_idx': self.idx,
            'val_auc': float(model.eval(val_data).split(':')[1]),
            'test_auc': float(model.eval(q_labeled).split(':')[1]),
            'rejection_rate': 1 - sample_size / self.sample_size,
            'test_probabilities': q_pseudo_probabilities if q_pseudo_probabilities is not None else model.predict(
                q_labeled),
            'count': sample_size,
            'seed': self._seed
        })
        self.idx += 1

    def freeze(self):
        self.record = self.get_record()

    def get_record(self):
        if isinstance(self.record, pd.DataFrame):
            return self.record
        else:
            return pd.DataFrame(self.record)

    def save(self, path):
        self.get_record().to_csv(path, index=False)

    @staticmethod
    def load(path):
        x = XGBDetectronRecord(sample_size=None)
        x.record = pd.read_csv(path)
        x.sample_size = x.record.query('ensemble_idx==0').iloc[0]['count']
        return x

    def counts(self, max_ensemble_size=None) -> np.ndarray:
        assert max_ensemble_size > 0 or max_ensemble_size is None, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size != -1:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['count'])
        return np.array(counts)

    def count_quantile(self, quantile, max_ensemble_size=None):
        counts = self.counts(max_ensemble_size)
        return np.quantile(counts, quantile, method='inverted_cdf')
