"""Utilities for post processing."""
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from nats_bench import create

import sys
sys.path.append('../src')

from kernels.weisfilerlehman import WeisfilerLehman
from bayes_quad.gp import GraphGP
from bayes_quad.quadrature import IntegrandModel
from utils.nas import index_to_nx


def get_dataset_from_collection(collection: str) -> str:
    dataset = collection.split('_')[1]
    if 'imagenet' in dataset:
        ds_name = 'ImageNet16-120'
    elif dataset == 'cifar10':
        ds_name = 'cifar10-valid'
    else:
        ds_name = 'cifar100'
    return ds_name


def sort_design_set_by_likelihood(archs, log_likelihoods, ensemble_weights):
    indices = np.argsort(log_likelihoods)[::-1]
    archs = [archs[i] for i in indices.tolist()]
    return (
        archs,
        np.array(log_likelihoods)[indices].tolist(),
        ensemble_weights[indices]
    )


def rebuild_surrogate(
    api, archs, log_likelihoods, normalise_y=False
) -> IntegrandModel:
    """Rebuild surrogate."""
    archs = [index_to_nx(api, a, hp="200") for a in archs]
    log_likelihoods = torch.tensor(log_likelihoods)
    log_offset = log_likelihoods.max()
    shifted_log_likelihoods = log_likelihoods - log_offset
    scaled_likelihoods = shifted_log_likelihoods.exp()
    kernel = [WeisfilerLehman(h=1)]
    surrogate_model = GraphGP(
        archs,
        scaled_likelihoods,
        kernel, 
        perform_y_normalization=normalise_y,
        log_offset=log_offset
    )
    integrand_model = IntegrandModel(surrogate=surrogate_model)
    surrogate_model.fit(wl_subtree_candidates=(1,), max_lik=1e-1)
    return integrand_model
