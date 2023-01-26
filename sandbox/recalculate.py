import os
import logging
from pathlib import Path
import numpy as np
import torch
from nats_bench import create
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
# from pp_utils import get_dataset_from_collection

import sys
sys.path.append('../src')

from model.arch_ensemble import ArchEnsemble
from utils.evaluation import TEST_BATCH_SIZE


logging.basicConfig(level=logging.INFO)

def run_post_processing(
    dataset: str,
    log_dir: int,
    warping: str = None,
    acquisition_strategy: str = 'uncertainty_sampling_sqrt',
    kernel_integration: str = 'random',
    truncate: str = None,
    truncate_sort: str = 'weight',
    truncate_level: int = None,
    collection_dir: Path = None
):
    if warping == 'None':
        warping = None
        sum_space = 'logit'
    else:
        sum_space = 'probability'
    truncate = None if truncate == 'None' else truncate
    truncate_level = int(truncate_level) if truncate_level is not None else None

    api = create(None, 'tss', fast_mode=True, verbose=False)

    # Evaluate with truncated ensembles.
    # Setup
    if kernel_integration == 'random':
        kernel_integration = {'num_samples': 2048}
    elif kernel_integration == 'exact':
        kernel_integration = {
            'num_samples': 15625, 'batch_size': 625, 'exact_sum': True
        }
    numerics = {
        'method': 'bayesian_quadrature',
        'surrogate': {'warping': warping},
        'initialisation': {'num_samples': 10},
        'evaluation_budget': 150,
        'acquisition': {
            'strategy': f'load_{acquisition_strategy}', 'load_id': log_dir
        },
        'kernel_integration': kernel_integration,
        'sum_space': sum_space,
        'nas_options': {'seed': 777, 'train_epochs': 200},
        'truncate': {'method': truncate, 'sort': truncate_sort, 'level': truncate_level},
        'test_batch_size': TEST_BATCH_SIZE[dataset]
    }
    log_dir_ = Path(f'../logs/')
    ensemble = ArchEnsemble(dataset, api, Path('../data'), numerics=numerics)

    def recalculated_ensemble(log_dir, sacred_run=None):
        # Truncate
        ensemble.learn(
            numerics,
            log_dir=log_dir,
            sacred_run=sacred_run,
            debug_options={'test_set': 'grid'}
        )
        # Evaluate
        ensemble.evaluate(numerics, log_dir=log_dir, sacred_run=sacred_run)

    # Create Sacred experiment.
    ex = Experiment()
    # Add observer to which experiment configuration and results will be added.
    ex.observers.append(
        MongoObserver(
            url='mongodb://bqnasUser:FU4EuM0z@mongo:27017/bqnas?authMechanism=SCRAM-SHA-1',
            db_name='bqnas'
        )
    )
    ex.observers.append(FileStorageObserver('../logs'))
    @ex.main
    def run_experiment(_config=None, _run=None):
        if collection_dir is not None:
            collection_dir.mkdir(parents=True, exist_ok=True)
            filename = collection_dir / 'run_ids.txt'
            append_write = 'a' if os.path.exists(filename) else 'w'
            file = open(filename, append_write)
            file.write(f'{_run._id}\n')
            file.close()
        recalculated_ensemble(log_dir_ / str(_run._id), sacred_run=_run)
    ex.run()
