from abc import ABC, abstractmethod
from copy import deepcopy
import functools
import logging
from pathlib import Path
import random
from typing import Sequence, Mapping, Union
import numpy as np
from networkx import Graph
import torch
from nats_bench import NATStopology

from bayes_quad.generate_test_graphs import create_nasbench201_graph
from utils.evaluation import evaluate_ensemble
from utils.nas import get_architecture_log_likelihood, get_data_splits, index_to_nx, rank_design_set, unique_random_samples, OPS


logger = logging.getLogger(__name__)


def neural_ensemble_search_regularised_evolution(
    api: NATStopology,
    dataset: str,
    nas_options: Mapping,
    data_dir: Path = Path('../'),
    evaluation_budget: int = 150,
    population_size: int = 50,
    num_parent_candidates: int = 10,
    ensemble_size: int = 3,
    load_design_set: int = None,
    log_dir: Path = Path('./'),
    smoke_test: bool = False
) -> Sequence[Graph]:
    """Neural Ensemble Search with Regularised Evolution for the 
    NATS-Bench topology search space.

    This is based on the implementation of
    https://github.com/automl/nes.

    :param evaluation_budget: The evaluation budget.
    :param population_size: The population size.
    :param num_parent_candidates: The number of parent candidates for
        evolution.
    :param ensemble_size: The ensemble size.
    :return: The architectures in the ensemble.
    """
    # Load the validation dataset.
    _, valid_loader, _, num_classes = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset,
        data_dir=data_dir, batch_size=nas_options.get('batch_size')
    )
    # Sample initial random population.
    if smoke_test:
        population_size = 4
        num_parent_candidates = 2
        ensemble_size = 2
    if load_design_set:
        with open(log_dir.parent / f'{load_design_set}/archs.txt') as f:
            population = f.readlines()
            print(f)
        population = list(map(lambda x: int(x[:-1]), population))
        population = [
            index_to_nx(api, a, hp=nas_options.get('train_epochs', "200"))
            for a in population
        ]
    else:
        population = unique_random_samples(
            population_size,
            api,
            dataset,
            **nas_options
        )
    query_arch = functools.partial(
        get_architecture_log_likelihood, api=api, dataset=dataset, data_dir=data_dir
    )
    log_likelihoods = torch.tensor([query_arch(a) for a in population])
    population, log_likelihoods = rank_design_set(population, log_likelihoods)
    # Perform regularised evolution.
    history = deepcopy(population)
    num_iterations = evaluation_budget - len(population)
    for i in range(num_iterations):
        logger.info(f'NES-RE Iteration {i + 1}/{num_iterations}')
        # Select the new parent.
        parent_candidates, _ = forward_selector(
            num_parent_candidates,
            population,
            log_likelihoods[-len(population):],
            valid_loader,
            num_classes,
            api,
            dataset,
            smoke_test=smoke_test,
            **nas_options
        )
        parent = random.choice(parent_candidates)
        # Mutate the parent.
        child = create_nasbench201_graph(mutate_arch(parent))
        population.append(child)
        history.append(child)
        log_likelihoods = torch.cat(
            (log_likelihoods, torch.tensor([query_arch(child)])), dim=0
        )
        # Maintain population size.
        if len(population) >= population_size:
            population.pop(0)
        if smoke_test and i > 0:
            break
    # Final selection.
    ensemble, ens_log_likelihoods = forward_selector(
        ensemble_size,
        history,
        log_likelihoods,
        valid_loader,
        num_classes,
        api,
        dataset,
        smoke_test=smoke_test,
        **nas_options
    )
    # Save the history and subset.
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(str(log_dir / 'archs.txt'), 'a') as f:
        for a in history:
            index = api.query_index_by_arch(a.name)
            f.write(f'{index}\n')
    with open(str(log_dir / 'archs_subset.txt'), 'a') as f:
        for a in ensemble:
            index = api.query_index_by_arch(a.name)
            f.write(f'{index}\n')
    return ensemble, ens_log_likelihoods


def forward_selector(
    ensemble_size: int,
    population: Sequence[Union[Graph, int]],
    log_likelihoods: torch.Tensor,
    valid_loader: torch.utils.data.DataLoader,
    num_classes: int,
    api: NATStopology,
    dataset: str,
    seed: int = 777,
    train_epochs: str = "200",
    smoke_test: bool = False,
    **kwargs
) -> Sequence[Graph]:
    """Forward selection from a population.

    :param ensemble_size: The ensemble size.
    :param population: The population from which to select.
    :return: The ensemble.
    """
    # Initialise with the best member.
    population = deepcopy(population)
    index = log_likelihoods.argmax().item()
    log_likelihoods = log_likelihoods.numpy().tolist()
    ensemble = [population.pop(index)]
    ells = [log_likelihoods.pop(index)]
    num_data = len(valid_loader.dataset) / 2  # Validation set is always half a DataLoader.
    current_size = len(ensemble)
    while current_size < ensemble_size:
        logger.info(f'Beam Search Member {current_size}/{ensemble_size}')
        losses = []
        for candidate in population:
            ens_log_likelihood, *_ = evaluate_ensemble(
                valid_loader,
                num_data,
                num_classes,
                ensemble + [candidate],
                api,
                dataset,
                split='valid',
                seed=seed,
                train_epochs=train_epochs,
                weights=torch.ones(len(ensemble) + 1) / (len(ensemble) + 1),
                param_ensemble_size=1,
                param_weights='even',
                sum_space='probability',
                smoke_test=smoke_test
            )
            losses.append(-ens_log_likelihood)
        index = torch.tensor(losses).argmin().item()
        ensemble.append(population.pop(index))
        ells.append(log_likelihoods.pop(index))
        current_size = len(ensemble)
    return ensemble, torch.tensor(ells)


def mutate_arch(architecture):
    """From NES repo
    # no hidden state mutation since the cell is fully connected
    # following https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py#L91
    # we also remove the 'identity' mutation 
    """
    mutation = np.random.choice(['op_mutation'])
    parent = deepcopy(architecture) # work with the list of ops only
    parent = list(filter(lambda i: (i != '' and i != '+'), parent.name.split('|')))
    parent = list(map(lambda x: x[:-2], parent))

    if mutation == 'identity':
        return parent
    elif mutation == 'op_mutation':
        edge_id = random.randint(0, len(parent)-1)
        edge_op = parent[edge_id]
        sampled_op = random.choice(OPS)
        while sampled_op == edge_op:
            sampled_op = random.choice(OPS)
        parent[edge_id] = sampled_op
        return parent
