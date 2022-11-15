import numpy as np

from data import sample_data
from data.core import split_dataset
from models import pretrained

model = pretrained.resnet18_trained_on_cifar10().model  # loads a resnet 18 model pretrained on cifar 10
model.eval()

p_train, p_val, p_test_all = sample_data.cifar10(split='all')  # loads the cifar 10 dataset
# above model was trained on p_train + p_val

q_all = sample_data.cifar10_1()  # loads the cifar 10_1 dataset

test_sets = {'p': p_test_all, 'q': q_all}


def compute_proj_norm(model, dataset):
    """ Takes a model and a dataset and computes the projection norm of the model on the dataset """
    return 0


if __name__ == '__main__':
    samples = [10, 20, 50]
    seeds = 100
    alpha = 0.05

    proj_norms = {x: {'p': [], 'q': []} for x in samples}
    for N in samples:
        for seed in range(seeds):
            for test in ['p', 'q']:
                q, _ = split_dataset(test_sets[test], random_seed=seed, num_samples=N)
                proj_norm_val = compute_proj_norm(model, q)
                print(f'N={N}, seed={seed}, test={test}, proj_norm={proj_norm_val}')
                proj_norms[N][test].append(proj_norm_val)

    # find quantile on in dist data
    for N in samples:
        thresh = np.quantile(proj_norms[N]['p'], alpha)
        power = (np.array(proj_norms[N]['q']) < thresh)
        print(f'N={N}, power={power.mean()} +- {power.std() / np.sqrt(seeds)}')
