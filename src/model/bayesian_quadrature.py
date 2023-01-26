from typing import Mapping
from pathlib import Path
import logging
import functools
from sacred.run import Run
import numpy as np
import torch
from nats_bench import NATStopology

from bayes_quad.generate_test_graphs import mutation
from bayes_quad.quadrature import (
    IntegrandModel, ExpIntegrandModel, SqIntegrandModel
)
from bayes_quad.gp import GraphGP, ExpGraphGP, SqGraphGP
from bayes_quad.acquisitions import (
    GraphUncertaintySampling, GraphExpectedImprovement
)
from bayes_quad.plotting import plot_kernel_inverse
from bayes_quad.recombination import (
    noise_until_positive, nystrom_test_functions, tchernychova_lyons_car, tchernychova_lyons_simplex
)
from bayes_quad.truncation import greedy_pairwise_reduction, likelihood_determinant_objective, reduce_partial_correlations
from kernels.weisfilerlehman import WeisfilerLehman
from model.deep_ensemble import learn_deep_ensembles
from model.neural_ensemble_search_regularised_evolution import neural_ensemble_search_regularised_evolution
from utils.nas import (
    get_data_splits,
    index_to_nx,
    prune_arches,
    random_sampling,
    load_design_set,
    get_architecture_log_likelihood,
    rank_design_set
)
from utils.evaluation import (
    get_test_metrics, get_test_archs, get_surrogate_performance_metrics
)
from utils.plotting import (
    plot_calibration, plot_calibration_comparison
)
from .plotting import plot_ensemble_weights


logger = logging.getLogger(__name__)


def bayesian_quadrature(
    api: NATStopology,
    dataset: str,
    surrogate: Mapping = None,
    initialisation: Mapping = None,
    evaluation_budget: int = 4,
    kernel_integration: Mapping = None,
    acquisition: Mapping = None,
    nas_options: Mapping = None,
    data_dir: Path = Path('../data'),
    log_dir: Path = Path('./logs'),
    sacred_run: Run = None,
    debug_options: Mapping = None,
    **kwargs
) -> None:
    """Perform Bayesian Quadrature to marginalise over
    architectures.

    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    :param surrogate: Surrogate options.
    :param initialisation: Initialisation options.
    :param kernel_integration: Kernel integration options.
    :param acquisition: Acquisition function options.
    :param nas_options: NAS options.
    :param data_dir: The data directory.
    :param log_dir: Logging directory.
    :param sacred_run: Sacred run object.
    :param debug_options: Options related to debugging.
        smoke_test: whether to run with fastest options.
        test_set: "random", "grid", or None. The test set size will
            always be 625. If "grid" then we select every 25th arch,
            ranked by (the average over seed=[777,888],
            train_epochs=200) validation loss.
    :param kwargs: Swallows redundant arguments.
    :return: The chosen architectures, and their corresponding weights.
    """
    nas_options = {} if nas_options is None else nas_options
    debug_options = {} if debug_options is None else debug_options
    nas_seed = nas_options.get('seed', None)
    train_epochs = nas_options.get('train_epochs', 200)
    swag = nas_options.get('swag', False)
    restrict_archs_to_best = nas_options.get(
        'restrict_archs_to_best', None
    )
    smoke_test = debug_options.get('smoke_test', False)
    initialisation = {} if initialisation is None else initialisation
    kernel_integration = (
        {} if kernel_integration is None else kernel_integration
    )
    surrogate_warping = surrogate.get('warping', None)
    acquisition_strategy = acquisition.get('strategy', 'random')
    if smoke_test:
        num_initial_samples = 2
    elif (
            acquisition_strategy == 'random'
            or acquisition_strategy=='top-valid'
    ):
        num_initial_samples = evaluation_budget
    else:
        num_initial_samples = initialisation.get('num_samples', 2)
    # Sample initial.
    if acquisition_strategy == 'top-valid':
        # run utils/nas.py to create pickle.
        pickle = np.load(
            f'../logs/sandbox/nas/{dataset}/ranked_architectures.npz'
        )
        locations = pickle['ranking'][:num_initial_samples]
        locations = [
            index_to_nx(api, a, hp=train_epochs) for a in locations
        ]
    elif 'load' in acquisition_strategy:
        locations, _, _ = load_design_set(
            api,
            dataset,
            log_dir.parent / str(acquisition['load_id']),
            hp=train_epochs,
            seed=nas_seed
        )
        locations = locations[:2] if smoke_test else locations
        locations = [
            index_to_nx(api, a, hp=train_epochs) for a in locations
        ]
    else:
        # locations, _, _ = random_sampling(
        #     pool_size=num_initial_samples,
        #     benchmark='nasbench201'  # NATS-Bench has same search space.
        # )
        locations = []
        while len(locations) < num_initial_samples:
            sample = random_sampling(
                1,
                api,
                dataset,
                hp=train_epochs,
                restrict_to_best=restrict_archs_to_best
            )
            try:
                if all(sample[0].name != l.name for l in locations):
                    locations += sample
            except (AttributeError, IndexError) as e:
                pass
    # Get the test set.
    test_locations = get_test_archs(
        debug_options.get('test_set', None),
        dataset,
        log_dir=log_dir.parent
    )
    # Query architecture log likelihoods.
    _, valid_loader, *_ = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset,
        data_dir=data_dir
    )
    batching_correction = (
        len(valid_loader.dataset) / valid_loader.batch_size
    )
    query_arch = functools.partial(
        get_architecture_log_likelihood,
        api=api,
        dataset=dataset,
        hp=train_epochs,
        seed=nas_seed,
        swag=swag,
        data_dir=data_dir,
        batching_correction=batching_correction
    )
    log_likelihoods = torch.tensor([query_arch(l) for l in locations])
    locations, log_likelihoods = rank_design_set(locations, log_likelihoods)
    log_offset = log_likelihoods.max()
    shifted_log_likelihoods = log_likelihoods - log_offset
    scaled_likelihoods = shifted_log_likelihoods.exp()
    if test_locations is not None:
        test_locations = [
            index_to_nx(api, a, hp=train_epochs)
            for a in test_locations
        ]
        test_locations = [a for a in test_locations if a is not None]
        test_log_likelihoods = torch.tensor(
            [query_arch(l) for l in test_locations]
        )

    # Set up kernel.
    kernel = [
        WeisfilerLehman(
            h=1, oa=surrogate.get('kernel', {}).get('oa', False)
        )
    ]

    if 'uncertainty_sampling' in acquisition_strategy:
        # Set up surrogate
        if 'log' in acquisition_strategy:
            surrogate_model = ExpGraphGP(
                locations,
                shifted_log_likelihoods,
                kernel,
                log_offset=log_offset
            )
        elif 'sqrt' in acquisition_strategy:
            surrogate_model = SqGraphGP(
                locations,
                shifted_log_likelihoods,
                kernel,
                log_offset=log_offset
            )
        else:
            raise NotImplementedError
        # Optimise surrogate.
        surrogate_model.fit(wl_subtree_candidates=(1,), max_lik=1e-1)
        _ = get_surrogate_performance_metrics(
            locations,
            log_likelihoods,
            surrogate_model,
            log_dir=log_dir,
            split='train',
            sacred_run=sacred_run
        )
        if test_locations is not None:
            _ = get_surrogate_performance_metrics(
                test_locations,
                test_log_likelihoods,
                surrogate_model,
                log_dir=log_dir,
                split='test',
                sacred_run=sacred_run
            )
        # Set up acquisition function
        acquisition_func = GraphUncertaintySampling(surrogate_model)
    elif acquisition_strategy == 'expected_improvement':
        # Set up surrogate
        surrogate_model = GraphGP(
            locations,
            scaled_likelihoods,
            kernel,
            perform_y_normalization=True,
            log_offset=log_offset
        )
        # Optimise surrogate.
        surrogate_model.fit(wl_subtree_candidates=(1,), max_lik=1e-1)
        _ = get_surrogate_performance_metrics(
            locations,
            log_likelihoods,
            surrogate_model,
            log_dir=log_dir,
            split='train',
            sacred_run=sacred_run
        )
        if test_locations is not None:
            _ = get_surrogate_performance_metrics(
                test_locations,
                test_log_likelihoods,
                surrogate_model,
                log_dir=log_dir,
                split='test',
                sacred_run=sacred_run
            )
        # Set up acquisition function
        acquisition_func = GraphExpectedImprovement(surrogate_model)

    if 'load' in acquisition_strategy or 'random' in acquisition_strategy:
        num_iterations = 0
    else:
        num_iterations = evaluation_budget - len(locations)

    # Make acquisitions
    for i in range(num_iterations):
        logger.info(f'Acquisition {i + 1}/{num_iterations}')
        # Make acquisition
        pool_size = 4 if smoke_test else acquisition.get('num_initial', 4)
        mutate_size = 2 if smoke_test else acquisition.get(
            'mutate_size', None
        )
        if mutate_size:
            if restrict_archs_to_best:
                raise NotImplementedError
            candidates, _ = mutation(
                locations,
                -shifted_log_likelihoods,  # Assumes smaller is better.
                n_best=acquisition.get(
                    'n_best', int(min(16, mutate_size / 2))
                ),
                n_mutate=mutate_size,
                pool_size=pool_size,
                patience=mutate_size,
                benchmark='nasbench201'
            )
            candidates = prune_arches(candidates)
        else:
            # candidates, _, _ = random_sampling(
            #     pool_size=pool_size,
            #     benchmark='nasbench201'
            # )
            candidates = random_sampling(
                pool_size,
                api,
                dataset,
                hp=train_epochs,
                restrict_to_best=restrict_archs_to_best
            )
        # Gradients not useful; no point in tracking them.
        with torch.no_grad():
            query, _, _ = acquisition_func.propose_location(
                candidates, top_n=1, queried=locations
            )
        query = query[0]
        query_log_likelihood = torch.tensor([query_arch(query)])
        logger.info(f'Selected: {query.name}')
        logger.info(f'Log Likelihood: {query_log_likelihood.item()}')
        for location in locations:
            assert location.name != query.name
        # Update datasets
        locations.append(query)
        log_likelihoods = torch.cat(
            (log_likelihoods, query_log_likelihood),
            dim=0
        )
        locations, log_likelihoods = rank_design_set(locations, log_likelihoods)
        log_offset = log_likelihoods.max()
        shifted_log_likelihoods = log_likelihoods - log_offset
        scaled_likelihoods = shifted_log_likelihoods.exp()
        # Re-optimise surrogate
        surrogate_model.reset_XY(
            locations, shifted_log_likelihoods, log_offset=log_offset
        )
        surrogate_model.fit(wl_subtree_candidates=(1,), max_lik=1e-1)
        _ = get_surrogate_performance_metrics(
            locations,
            log_likelihoods,
            surrogate_model,
            log_dir=log_dir,
            split='train',
            sacred_run=sacred_run
        )
        if test_locations is not None:
            _ = get_surrogate_performance_metrics(
                test_locations,
                test_log_likelihoods,
                surrogate_model,
                log_dir=log_dir,
                split='test',
                sacred_run=sacred_run
            )
        if smoke_test and i > 0:
            break
    # Compute quadrature weights.
    if surrogate_warping is None:
        surrogate_model = GraphGP(
            locations,
            scaled_likelihoods,
            kernel, 
            perform_y_normalization=surrogate.get('normalise_y', False),
            log_offset=log_offset
        )
        integrand_model = IntegrandModel(surrogate=surrogate_model)
        surrogate_model.fit(wl_subtree_candidates=(1,), max_lik=1e-1)
        _ = get_surrogate_performance_metrics(
            locations,
            log_likelihoods,
            surrogate_model,
            log_dir=log_dir,
            split='train',
            sacred_run=sacred_run
        )
        if test_locations is not None:
            _ = get_surrogate_performance_metrics(
                test_locations,
                test_log_likelihoods,
                surrogate_model,
                log_dir=log_dir,
                split='test',
                sacred_run=sacred_run
            )
    elif surrogate_warping == 'moment-matched-log':
        integrand_model = ExpIntegrandModel(surrogate=surrogate_model)
    elif surrogate_warping == 'linearised-sqrt':
        integrand_model = SqIntegrandModel(surrogate=surrogate_model)
    log_likelihoods = log_likelihoods
    with torch.no_grad():
        quadrature_weights = integrand_model.compute_quadrature_weights(
            **kernel_integration,
            api=api,
            dataset=dataset,
            restrict_archs_to_best=restrict_archs_to_best,
            smoke_test=smoke_test,
            log_dir=log_dir
        )
        log_model_evidence = integrand_model.posterior(
            weights=quadrature_weights
        ).loc.log() + log_offset
    logger.info(f'Log Model Evidence: {log_model_evidence}')
    if sacred_run is not None:
        sacred_run.log_scalar(
            metric_name='log_model_evidence',
            value=log_model_evidence.item()
        )
    # Save relevant quantities.
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(str(log_dir / 'archs.txt'), 'a') as f:
        for a in locations:
            index = api.query_index_by_arch(a.name)
            f.write(f'{index}\n')
    plot_kernel_inverse(integrand_model, log_dir=log_dir)
    if surrogate_warping is None:
        ensemble_weights = quadrature_weights * (
            log_likelihoods - log_model_evidence
        ).exp()
        if torch.isnan(ensemble_weights).any():
            logger.warning('Model evidence estimate is negative.')
            ensemble_weights = (
                quadrature_weights * scaled_likelihoods / (
                    quadrature_weights @ integrand_model.surrogate.y
                    + integrand_model.surrogate.y_mean
                )
            )
        np.save(
            str(log_dir / 'ensemble-weights.npy'),
            ensemble_weights.numpy()
        )
        plot_ensemble_weights(
            log_likelihoods.numpy(),
            ensemble_weights.numpy(),
            log_dir
        )
    elif surrogate_warping == 'moment-matched-log':
        ensemble_weights = quadrature_weights
        np.savez(
            str(log_dir / 'ensemble-weights.npz'),
            architecture_weights=ensemble_weights[
                'architecture_weights'
            ],
            sample_weights=ensemble_weights['sample_weights']
        )
    elif surrogate_warping == 'linearised-sqrt':
        ensemble_weights = quadrature_weights
        np.save(
            str(log_dir / 'ensemble-weights.npy'),
            ensemble_weights.numpy()
        )
    
    return (
        locations,
        log_likelihoods,
        ensemble_weights,
        integrand_model,
        quadrature_weights,
        log_model_evidence,
    )
