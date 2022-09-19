import pytorch_lightning as pl
import torch
from utils.detectron import DetectronLoader, DetectronModule, EarlyStopper

from tests.detectron.detectron import infer_labels
from data import sample_data
from data.core import split_dataset
from models import pretrained

load_model = lambda: pretrained.resnet18_trained_on_cifar10()
p_train, p_val, p_test_all = sample_data.cifar10(split='all')
q_all = sample_data.cifar10_1()

test_sets = {'p': p_test_all, 'q': q_all}
base_model = load_model()

# hyperparams ---------------------------------------------
max_epochs_per_model = 5
optimizer = lambda params: torch.optim.Adam(params, lr=1e-3)
ensemble_size = 10
batch_size = 512
patience = 2
# ---------------------------------------------------------
runs = 100
# ---------------------------------------------------------
gpus = [0]
num_workers = 12
# ---------------------------------------------------------

(pseudo_labels_train, _), (pseudo_labels_val, val_acc) = infer_labels(
    model=base_model,
    dataset=(p_train, p_val),
    gpus=gpus,
    batch_size=batch_size,
    num_workers=num_workers,
    verbose=True,
    return_accuracy=True,
)

for N in [10, 20, 50]:
    for dataset_name in ['p', 'q']:
        for seed in range(runs):
            pl.seed_everything(seed)
            val_results = []
            test_results = []
            log = {'N': N, 'seed': seed, 'dataset': dataset_name, 'ensemble_idx': 0}
            count = N
            q, _ = split_dataset(test_sets[dataset_name], N, seed)
            pseudo_labels_test = infer_labels(
                model=base_model,
                dataset=q,
                gpus=gpus,
                batch_size=N,
                num_workers=num_workers,
                verbose=True
            )

            pq_loader = DetectronLoader(p_train=p_train,
                                        p_val=p_val,
                                        q=q,
                                        p_train_pseudo_labels=pseudo_labels_train,
                                        p_val_pseudo_labels=pseudo_labels_val,
                                        q_pseudo_labels=pseudo_labels_test,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        )

            base = DetectronModule(base_model)
            pl.Trainer(gpus=gpus, logger=False, max_epochs=1).test(base, pq_loader.test_dataloader(), verbose=False)
            test_results.append(base.test_struct.to_dict() | {'count': count} | log)
            val_results.append({'accuracy': val_acc, 'rejection_rate': 0, 'accepted_accuracy': val_acc} | log)
            stopper = EarlyStopper(patience=patience, mode='min')
            stopper.update(count)

            for i in range(1, ensemble_size + 1):
                log.update({'ensemble_idx': i})

                trainer = pl.Trainer(
                    gpus=gpus,
                    max_epochs=max_epochs_per_model,
                    logger=False,
                    num_sanity_val_steps=0,
                    limit_val_batches=0,
                    enable_model_summary=False
                )
                alpha = 1 / (len(pq_loader.train_dataloader()) * count + 1)
                detector = DetectronModule(model=load_model(),
                                           alpha=alpha)
                print(f'α = {1000 * alpha:.3f} × 10⁻³')
                trainer.fit(detector, pq_loader)
                trainer.test(detector, pq_loader.val_dataloader(), verbose=False)
                val_results.append(detector.test_struct.to_dict(minimal=True) | log)

                trainer.test(detector, pq_loader.test_dataloader(), verbose=False)
                count = pq_loader.refine(~detector.test_struct.rejection_mask, verbose=True)
                test_results.append(detector.test_struct.to_dict() | {'count': count} | log)

                if stopper.update(count):
                    print('Early stopping after', i, 'iterations')
                    break

                if count == 0:
                    print(f'Converged to rejection rate of 100% after {i} iterations')
                    break

            torch.save(val_results, f'results/cifar/val_{seed}_{dataset_name}_{N}.pt')
            torch.save(test_results, f'results/cifar/test_{seed}_{dataset_name}_{N}.pt')
