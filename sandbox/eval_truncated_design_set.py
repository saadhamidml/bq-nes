
import functools
import os
import logging
from pathlib import Path
import numpy as np
import torch
from nats_bench import create
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
# from pp_utils import get_dataset_from_collection

import sys; sys.path.append('../src')

from utils.nas import get_architecture_log_likelihood, get_data_splits, index_to_nx, load_design_set
from model.arch_ensemble import ArchEnsemble
from utils.evaluation import TEST_BATCH_SIZE

from sandbox_utils import rebuild_surrogate

logging.basicConfig(level=logging.INFO)

def run_post_processing(
    dataset: str,
    run_id: int,
    sum_space: str = 'probability',
    truncate_method: str = 'positive_weights',
    truncate_sort: str = 'weight',
    truncate_level: int = None,
    dataset_split: str = 'valid',
    log_dir: Path = Path('../logs'),
    data_dir: Path = Path('../data'),
    collection_dir: Path = None
):
    log_dir /= str(run_id)
    truncate_method = None if truncate_method == 'None' else truncate_method
    try:
        truncate_level = int(truncate_level)
    except Exception as e:
        truncate_level = None

    api = create(None, 'tss', fast_mode=True, verbose=False)
    archs, _, ensemble_weights = load_design_set(
        api, dataset, log_dir
    )

    # pickle = np.load(f'../logs/sandbox/nas/{dataset}/ranked_architectures.npz')
    # ranking = pickle['ranking'][:3]

    # for r in ranking:
    #     if r not in archs:
    #         archs.append(int(r))

    ensemble_weights = torch.tensor(ensemble_weights.tolist())
    train_loader, valid_loader, *_ = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset,
        data_dir=data_dir
    )
    split_loader = valid_loader if dataset_split == 'valid' else train_loader
    batching_correction = (
        len(split_loader.dataset) / split_loader.batch_size
    )
    query_arch = functools.partial(
        get_architecture_log_likelihood,
        api=api,
        dataset=dataset,
        dataset_split=dataset_split,
        hp="200",
        seed=777,
        swag=False,
        data_dir=data_dir,
        batching_correction=batching_correction
    )
    log_likelihoods = torch.tensor([query_arch(l) for l in archs])
    numerics = {
        'method': 'bayesian_quadrature',
        'surrogate': {'warping': None},
        'sum_space': sum_space,
        'nas_options': {'seed': 777, 'train_epochs': 200},
        'truncate': {
            'method': truncate_method,
            'sort': truncate_sort,
            'level': truncate_level,
            'kernel_integration': 'load',
        },
        'test_batch_size': TEST_BATCH_SIZE[dataset]
    }
    log_dir_ = Path(f'../logs/')
    ensemble = ArchEnsemble(dataset, api, Path('../data'), numerics=numerics)
    
    integrand_model = rebuild_surrogate(
        api, archs, log_likelihoods, normalise_y=False
    )
    archs = integrand_model.surrogate.x
    try:
        int_kernel = torch.tensor(np.load(log_dir / 'kernel-integral.npy').tolist())
        # int_kernel = integrand_model.compute_kernel_sums(api=api, dataset=dataset)
        quadrature_weights = (
            int_kernel.view(1, -1) @ integrand_model.surrogate.K_i
        ).view(-1)
        # log_model_evidence = integrand_model.posterior(
        #     weights=quadrature_weights
        # ).loc.log() + integrand_model.surrogate.log_offset

        # ensemble_weights = quadrature_weights * (
        #     log_likelihoods - log_model_evidence
        # ).exp()
        # if torch.isnan(ensemble_weights).any():
        #     log_offset = log_likelihoods.max()
        #     shifted_log_likelihoods = log_likelihoods - log_offset
        #     scaled_likelihoods = shifted_log_likelihoods.exp()
        #     ensemble_weights = (
        #         quadrature_weights * scaled_likelihoods / (
        #             quadrature_weights @ integrand_model.surrogate.y
        #             + integrand_model.surrogate.y_mean
        #         )
        #     )
    except FileNotFoundError as e:
        int_kernel = None
        quadrature_weights = None

    log_model_evidence = torch.tensor({
        'cifar10-valid': -83.65742457546568,
        'cifar100': -50.04335642747517,
        'ImageNet16-120': -53.36738665227185
    }[dataset])

    ensemble._archs = archs
    ensemble._log_likelihoods = log_likelihoods
    ensemble._quadrature_weights = quadrature_weights
    ensemble._log_model_evidence = log_model_evidence
    ensemble._integrand_model = integrand_model
    ensemble._ensemble_weights = ensemble_weights

    ensemble._kernel_integrals = int_kernel

    def recalculated_ensemble(log_dir, sacred_run=None):
        ensemble.evaluate(
            None,  # (archs, log_likelihoods, ensemble_weights),
            numerics,
            log_dir=log_dir,
            sacred_run=sacred_run
        )

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
