from __future__ import annotations

from typing import Callable, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

from utils.generic import vprint, not_in_or_none
from utils.inference import LabelCollector

from utils.detectron import DetectronLoader, DetectronModule, DetectronEnsemble, EarlyStopper


def detectron_tst(p_train: Dataset,
                  p_val: Dataset,
                  q: Dataset,
                  base_model: torch.nn.Module,
                  create_detector: Callable[[], torch.nn.Module],
                  batch_size: int = 512,
                  ensemble_size=10,
                  max_epochs_per_model=4,
                  init_metric_val=None,
                  patience=2,
                  gpus=[1],
                  num_workers=16,
                  verbose=True,
                  pseudo_labels: Optional[dict[str, torch.Tensor]] = None,
                  **trainer_kwargs):

    val_results = []
    test_results = []

    count = len(q)
    q_pseudo_labels = get_pseudo_labels(pseudo_labels, q, 'q',
                                        base_model, batch_size, gpus,
                                        num_workers, verbose)

    p_train_pseudo_labels = get_pseudo_labels(pseudo_labels, p_train, 'p_train',
                                              base_model, batch_size, gpus,
                                              num_workers, verbose)

    p_val_pseudo_labels = get_pseudo_labels(pseudo_labels, p_val, 'p_val',
                                            base_model, batch_size, gpus,
                                            num_workers, verbose)

    pq_loader = DetectronLoader(p_train=p_train,
                                p_val=p_val,
                                q=q,
                                p_train_pseudo_labels=p_train_pseudo_labels,
                                p_val_pseudo_labels=p_val_pseudo_labels,
                                q_pseudo_labels=q_pseudo_labels,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                )

    base = DetectronModule(base_model)
    pl.Trainer(gpus=gpus, logger=False, max_epochs=1).test(base, pq_loader.test_dataloader(), verbose=False)
    test_results.append(base.test_struct.to_dict() | {'count': count})
    val_results.append({'accuracy': init_metric_val, 'rejection_rate': 0, 'accepted_accuracy': init_metric_val})
    stopper = EarlyStopper(patience=patience, mode='min')
    stopper.update(count)

    for i in range(1, ensemble_size + 1):

        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=max_epochs_per_model,
            logger=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            enable_model_summary=False,
            **trainer_kwargs
        )

        alpha = 1 / (len(pq_loader.train_dataloader()) * count + 1)
        detector = DetectronModule(model=create_detector(),
                                   alpha=alpha)
        print(f'α = {1000 * alpha:.3f} × 10⁻³')
        trainer.fit(detector, pq_loader)
        trainer.test(detector, pq_loader.val_dataloader(), verbose=False)
        val_results.append(detector.test_struct.to_dict(minimal=True))

        trainer.test(detector, pq_loader.test_dataloader(), verbose=False)
        count = pq_loader.refine(~detector.test_struct.rejection_mask, verbose=verbose)
        test_results.append(detector.test_struct.to_dict() | {'count': count})

        if stopper.update(count):
            vprint(f'Early stopping after {i} iterations')
            break

        if count == 0:
            vprint(f'Converged to rejection rate of 100% after {i} iterations')
            break

        return val_results, test_results


def get_pseudo_labels(pseudo_labels_dict: Optional[dict[str, torch.Tensor]],
                      dataset: Dataset,
                      dataset_key: str,
                      model: torch.nn.Module,
                      batch_size: Optional[int] = None,
                      gpus: Optional[list[int]] = None,
                      num_workers=12, verbose=True):
    if not_in_or_none(pseudo_labels_dict, dataset_key):
        return infer_labels(
            model=model,
            dataset=dataset,
            gpus=gpus,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose
        )
    else:
        return pseudo_labels_dict[dataset_key]


def infer_labels(model: torch.nn.Module, dataset: Dataset | Sequence[Dataset], batch_size: Optional[int] = None,
                 num_workers=64, gpus=[0],
                 verbose=True, return_accuracy=False):
    tr = pl.Trainer(gpus=gpus, max_epochs=1, enable_model_summary=False, logger=False)
    if isinstance(dataset, Dataset):
        dataset = [dataset]

    results = []
    for d in dataset:
        dl = DataLoader(d, batch_size=batch_size if batch_size else len(dataset),
                        num_workers=num_workers,
                        drop_last=False)

        lc = LabelCollector(model=model)
        tr.validate(lc, dl, verbose=False)
        if verbose:
            print(f'Inferred labels for {len(d)} samples. Accuracy: {lc.compute_accuracy():.3f}')
        results.append(lc.get_labels(mode='predicted'))
        if return_accuracy:
            results[-1] = [results[-1], (lc.compute_accuracy())]

    if len(dataset) == 1:
        if return_accuracy:
            return results[0][0], results[0][1]
        return results[0]
    return results
