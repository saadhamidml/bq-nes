from typing import Dict, Sequence
from itertools import count
from pathlib import Path
import numpy as np
from nats_bench import NATStopology
import torch
from xautodl.models import get_cell_based_tiny_net

from utils.nas import get_architecture_log_likelihood, get_data_splits, unique_random_samples
from .neural_network import train_network


SEEDS = count(start=999, step=111)


def select_architecture(
    evaluation_budget: int,
    api: NATStopology,
    dataset: str,
    seed: int = 777,
    train_epochs: str = "200",
    data_dir: Path = Path('../data')
) -> int:
    """Randomly sample architectures from the search space, and return
    the best one.

    :param evaluation_budget: Number to of architectures to sample.
    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    :return: The api index of the selected architecture.
    """
    archs = unique_random_samples(evaluation_budget, api, dataset, hp=train_epochs)
    log_likelihoods = [
        get_architecture_log_likelihood(
            a, api, dataset, seed=seed, hp=train_epochs, data_dir=data_dir
        ) for a in archs
    ]
    index = np.argmax(log_likelihoods)
    return api.query_index_by_arch(archs[index].name)


def learn_deep_ensembles(
    evaluation_budget: int,
    ensemble_size: int,
    api: NATStopology,
    dataset: str,
    data_dir: Path = Path('../data'),
    smoke_test: bool = False
) -> Sequence[int]:
    """Optimise architecture weights for the members of a DeepEnsemble.
    
    :param evaluation_budget: Number of architectures to randomly sample
        to select the architecture to make an ensemble of.
    :param ensemble_size: The ensemble size.
    :param dataset: The dataset.
    :param data_dir: The data directory.
    :param smoke_test: Whether to run with fastest settings (for
        debugging).
    :return: The API index of the chosen architecture.
    """
    # pickle = np.load(data_dir / f'rankings/{dataset}/valid_ranked_architectures.npz')
    # api_index = int(pickle['ranking'][0])
    api_index = select_architecture(evaluation_budget, api, dataset, data_dir=data_dir)
    config = api.get_net_config(api_index, dataset)
    network = get_cell_based_tiny_net(config)
    train_loader, *_ = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset, data_dir=data_dir
    )
    num_additional = ensemble_size - 2
    for _, seed in zip(range(num_additional), SEEDS):
        path = Path(f'{data_dir}/additional_networks/{dataset}/{api_index}/{seed}.pt')
        if smoke_test:
            path = path.with_name(f'{seed}_SMOKE_TEST.pt')
        if path.is_file():
            continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        train_network(network, train_loader=train_loader, seed=seed, smoke_test=smoke_test)
        torch.save(network.state_dict(), path)
    return [api_index]


def load_deep_ensemble(
    ensemble_size: int, api_index: int, dataset: str, data_dir: Path = Path('../')
) -> Sequence[Dict]:
    """Load the optimised architecture weights for a DeepEnsemble.
    
    :param ensemble_size: The ensemble size.
    :param dataset: The dataset.
    :param data_dir: The data directory.
    :return: The loaded parameter settings. Length is ensemble_size - 2,
        as the other two are provided by NATS-Bench.
    """
    num_additional = ensemble_size - 2
    state_dicts = {}
    for _, seed in zip(range(num_additional), SEEDS):
        path = Path(f'{data_dir}/additional_networks/{dataset}/{api_index}/{seed}')
        state_dicts[seed] = torch.load(path)
    return state_dicts
