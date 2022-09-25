from data.sample_data.cifar import cifar10_features
import xgboost as xgb


def train_booster():
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
        'verbosity': 2,
        'seed': 0
    }
    dtrain = xgb.DMatrix(*cifar10_features('train'))
    dval = xgb.DMatrix(*cifar10_features('val'))
    bst = xgb.train(DEFAULT_PARAMS, dtrain,
                    num_boost_round=50, evals=[(dval, 'val')],
                    early_stopping_rounds=10)
    bst.save_model('/voyager/datasets/cifar10_features/cifar10_features.model')
