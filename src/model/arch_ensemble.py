from copy import deepcopy
import json
from syslog import LOG_LOCAL0
from typing import Sequence, Union, Mapping, Tuple
from pathlib import Path
import logging
import warnings
import functools
from sacred.run import Run
from nats_bench.api_topology import NATStopology
from nats_bench.api_size import NATSsize
from networkx import Graph
import numpy as np
import math
import scipy.special
import pandas as pd
import torch
import xautodl
from PIL import Image
from xautodl.datasets import get_datasets
from xautodl.models import get_cell_based_tiny_net

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# from bayes_quad.generate_test_graphs import random_sampling, mutation
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
    noise_until_positive,
    nystrom_test_functions,
    optimise_weights,
    tchernychova_lyons_car,
    tchernychova_lyons_simplex,
    optimise_weights_iterative,
    optimise_weights_wce
)
from bayes_quad.truncation import greedy_pairwise_reduction, likelihood_determinant_objective, reduce_partial_correlations
from kernels.weisfilerlehman import WeisfilerLehman
from model.bayesian_quadrature import bayesian_quadrature
from model.deep_ensemble import learn_deep_ensembles
from model.neural_ensemble_search_regularised_evolution import neural_ensemble_search_regularised_evolution
from .neural_ensemble_search_bayesian_sampling import neural_ensemble_search_bayesian_sampling
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
    evaluate_ensemble, get_test_metrics, get_test_archs, get_surrogate_performance_metrics
)
from utils.plotting import (
    plot_calibration, plot_calibration_comparison
)
from .plotting import plot_ensemble_weights

logger = logging.getLogger(__name__)

class ArchEnsemble(torch.nn.Module):
    """An ensemble of architectures."""

    def __init__(
        self,
        dataset: str,
        api: Union[NATStopology, NATSsize],
        data_dir: Path,
        numerics: Mapping = None
    ) -> None:
        """Initilise ensemble.
        
        :param dataset: The dataset which is to be modelled. Options are
            'cifar10', 'cifar100', and 'ImageNet16-120'.
        :param api: The nats_bench api for either the topology or size
            search space.
        :param data_dir: Path of datasets.
        :param numerics: Numerics options.
        """
        super().__init__()
        self.dataset = dataset
        self.api = api
        self.data_dir = data_dir
        self._archs = None
        self._log_likelihoods = None
        self._kernel_integrals = None
        self._quadrature_weights = None
        self._log_model_evidence = None
        self._integrand_model = None
        self._ensemble_weights = None
        self._ensemble_indices = None
        # Only used to decide how to truncate the ensemble.
        self._numerics = {} if numerics is None else numerics
    
    @property
    def ensemble_indices(self):
        if self._ensemble_indices is not None:
            return self._ensemble_indices
        if self._archs is None:
            raise RuntimeError('Run ArchEnsemble.learn first!')
        options = self._numerics.get('truncate', None)
        if not options or options.get('method', None) is None:
            return torch.tensor(list(range(len(self._archs))))
        elif options['method'] == 'highest_likelihoods':
            return torch.argsort(self._log_likelihoods)[-options['level']:]
        elif options['method'] == 'positive_weights':
            mask = self._ensemble_weights > 0
            sortby = options.get('sort', 'weight')
            level = options.get('level')
            level = level if level is not None else mask.sum()
            if sortby == 'weight':
                to_sort = self._ensemble_weights
            elif sortby == 'likelihood':
                to_sort = self._log_likelihoods
            masked_inds = torch.argsort(to_sort[mask])[-level:]
            return torch.arange(self._ensemble_weights.shape[0])[mask][
                masked_inds
            ]
        elif options['method'] == 'nystrom_recombination':
            level = options.get('level')
            if level is None:
                level = self._log_likelihoods.size(0)
            test_functions, _ = nystrom_test_functions(
                self.api,
                self.dataset,
                self._numerics['nas_options']['train_epochs'],
                self._archs,
                self._integrand_model.surrogate,
                level=level - 2,
                return_integrals=False
            )
            # weights, indices = tchernychova_lyons_simplex(
            #     test_functions[mask][:-1], full_ints[mask][:-1]
            # )
            weights, indices = tchernychova_lyons_car(
                torch.flip(test_functions, dims=(1,)).T,
                torch.ones_like(self._quadrature_weights) / len(self._quadrature_weights)
            )
            indices = len(self._quadrature_weights) - indices - 1
            # Rescale the weights such that we get the same estimate for
            # the model evidence.
            log_wls = weights.log() + self._log_likelihoods[indices]
            log_me = torch.logsumexp(log_wls, dim=0)
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = (log_wls - log_me).exp()
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'posterior_recombination':
            level = options.get('level')
            if level is None:
                level = self._log_likelihoods.size(0)
            test_functions, _ = nystrom_test_functions(
                self.api,
                self.dataset,
                self._numerics['nas_options']['train_epochs'],
                self._archs,
                self._integrand_model.surrogate,
                level=level - 2,
                return_integrals=False
            )
            posterior = self._log_likelihoods - math.log(15625) - self._log_model_evidence
            posterior_ = (posterior - posterior.logsumexp(0)).exp()
            weights, indices = tchernychova_lyons_car(
                torch.flip(test_functions, dims=(1,)).T, torch.flip(posterior_, dims=(0,))
            )
            indices = len(self._quadrature_weights) - indices - 1
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'posterior_recombination_optimise':
            level = options.get('level')
            if level is None:
                level = self._log_likelihoods.size(0)
            test_functions, _ = nystrom_test_functions(
                self.api,
                self.dataset,
                self._numerics['nas_options']['train_epochs'],
                self._archs,
                self._integrand_model.surrogate,
                level=level - 2,
                return_integrals=False
            )
            posterior = self._log_likelihoods - math.log(15625) - self._log_model_evidence
            posterior_ = (posterior - posterior.logsumexp(0)).exp()
            weights, indices = tchernychova_lyons_car(
                torch.flip(test_functions, dims=(1,)).T, torch.flip(posterior_, dims=(0,))
            )
            indices = len(self._quadrature_weights) - indices - 1
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            weights = optimise_weights(
                self._archs,
                self._log_likelihoods,
                self.api,
                self.dataset,
                sum_space=self._numerics.get('sum_space', 'probability'),
                nas_options=self._numerics.get('nas_options', {}),
                data_dir=self.data_dir,
                weights=weights
            )
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'posterior_recombination_optimise_wce':
            level = options.get('level')
            if level is None:
                level = self._log_likelihoods.size(0)
            test_functions, _ = nystrom_test_functions(
                self.api,
                self.dataset,
                self._numerics['nas_options']['train_epochs'],
                self._archs,
                self._integrand_model.surrogate,
                level=level - 2,
                return_integrals=False
            )
            posterior = self._log_likelihoods - math.log(15625) - self._log_model_evidence
            posterior_ = (posterior - posterior.logsumexp(0)).exp()
            weights, indices = tchernychova_lyons_car(
                torch.flip(test_functions, dims=(1,)).T, torch.flip(posterior_, dims=(0,))
            )
            indices = len(self._quadrature_weights) - indices - 1
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            weights = optimise_weights_wce(
                self._archs,
                self._log_likelihoods,
                self._integrand_model.surrogate.K[indices][:, indices],
                self._kernel_integrals[indices],
                weights=weights
            )
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'likelihood_determinant_objective':
            level = options.get('level')
            indices = likelihood_determinant_objective(
                self._log_likelihoods, self._integrand_model.surrogate.K, level
            )
            self._ensemble_indices = indices
            return self._ensemble_indices
        elif options['method'] == 'greedy_pairwise_reduction':
            level = options.get('level')
            indices = greedy_pairwise_reduction(
                self._log_likelihoods, self._integrand_model.surrogate.K, level
            )
            archs = [self._archs[i] for i in indices]
            log_likelihoods = self._log_likelihoods[indices]
            test_functions, test_integrals = nystrom_test_functions(
                self.api,
                self.dataset,
                self._numerics['nas_options']['train_epochs'],
                self._archs,
                self._integrand_model.surrogate,
                level=level,
                return_integrals=False
            )
            test_functions = torch.cat(
                (test_functions[:-1, indices], log_likelihoods.exp().view(1, -1)), dim=0
            )
            test_integrals = torch.cat(
                (test_integrals[:-1], self._log_model_evidence.exp().view(-1)),
                dim=0
            )
            weights, rec_indices = tchernychova_lyons_simplex(test_functions, test_integrals)
            assert len(rec_indices) == len(indices)
            self._archs = [archs[i] for i in indices]
            self._log_likelihoods = log_likelihoods[indices]
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'reduce_partial_correlations':
            level = options.get('level', None)
            level = len(self._archs) if level is None else level
            indices = reduce_partial_correlations(
                self._log_likelihoods,
                self._integrand_model.surrogate.K,
                level
            )
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = noise_until_positive(
                self._kernel_integrals[indices],
                self._integrand_model.surrogate.K[indices, :][:, indices]
            )
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif options['method'] == 'noise_until_positive':
            self._ensemble_weights = noise_until_positive(
                self._kernel_integrals,
                self._integrand_model.surrogate.K
            )
            mask = self._ensemble_weights > 0
            sortby = options.get('sort', 'weight')
            level = options.get('level')
            level = level if level is not None else mask.sum()
            if sortby == 'weight':
                to_sort = self._ensemble_weights
            elif sortby == 'likelihood':
                to_sort = self._log_likelihoods
            masked_inds = torch.argsort(to_sort[mask])[-level:]
            return torch.arange(self._ensemble_weights.shape[0])[mask][
                masked_inds
            ]
        elif options['method'] == 'uncertainty_sampling_sqrt':
            # Select initial points as best likelihood and median likelihood.
            argmax = torch.argmax(self._log_likelihoods)
            argmedian = torch.median(self._log_likelihoods, dim=0)[1]
            locations = [self._archs[argmax], self._archs[argmedian]]
            log_likelihoods = torch.tensor([
                self._log_likelihoods[argmax],
                self._log_likelihoods[argmedian]
            ])
            _, valid_loader, *_ = get_data_splits(
                'cifar10' if self.dataset == 'cifar10-valid' else self.dataset,
                data_dir=self.data_dir
            )
            batching_correction = (
                len(valid_loader.dataset) / valid_loader.batch_size
            )
            query_arch = functools.partial(
                get_architecture_log_likelihood,
                api=self.api,
                dataset=self.dataset,
                hp=self.numerics.get('nas_options', {}).get(
                    'train_epochs', '200'
                ),
                seed=self.numerics.get('nas_options', {}).get('nas_seed', 777),
                swag=self.numerics.get('nas_options', {}).get('swag', False),
                data_dir=self.data_dir,
                batching_correction=batching_correction
            )
            log_offset = log_likelihoods.max()
            shifted_log_likelihoods = log_likelihoods - log_offset
            scaled_likelihoods = shifted_log_likelihoods.exp()
            kernel = [WeisfilerLehman(h=1)]
            surrogate_model = SqGraphGP(
                locations,
                shifted_log_likelihoods,
                kernel,
                log_offset=log_offset
            )
            with torch.enable_grad():
                surrogate_model.fit(wl_subtree_candidates=(1,))
            acquisition_func = GraphUncertaintySampling(surrogate_model)
            # Run acquisition loop.
            for i in range(options['level'] - len(locations)):
                logger.info(f'Truncation Acquisition {i + 1}')
                candidates = self._archs
                with torch.no_grad():
                    query, _, _ = acquisition_func.propose_location(
                        candidates, top_n=1
                    )
                query = query[0]
                query_log_likelihood = torch.tensor([query_arch(query)])
                locations.append(query)
                log_likelihoods = torch.cat(
                    (log_likelihoods, query_log_likelihood),
                    dim=0
                )
                log_offset = log_likelihoods.max()
                shifted_log_likelihoods = log_likelihoods - log_offset
                scaled_likelihoods = shifted_log_likelihoods.exp()
                surrogate_model.reset_XY(
                    locations, shifted_log_likelihoods, log_offset=log_offset
                )
                with torch.enable_grad():
                    surrogate_model.fit(wl_subtree_candidates=(1,))
            surrogate_model = GraphGP(locations, scaled_likelihoods, kernel)
            integrand_model = IntegrandModel(surrogate=surrogate_model)
            with torch.enable_grad():
                surrogate_model.fit(wl_subtree_candidates=(1,))
            self._archs = locations
            self._log_likelihoods = log_likelihoods
            with torch.no_grad():
                self._quadrature_weights = integrand_model.compute_quadrature_weights()
                self._log_model_evidence = integrand_model.posterior(
                    weights=self._quadrature_weights
                ).loc.log() + log_offset
            self._integrand_model = integrand_model
            self._ensemble_weights = self._quadrature_weights * (
                self._log_likelihoods - self._log_model_evidence
            ).exp()
            self._ensemble_indices = torch.tensor(list(range(len(self._archs))))
            return self._ensemble_indices
        elif 'greedy_negative_elimination' in options['method']:
            sort_inds = torch.argsort(self._log_likelihoods)
            archs = [index_to_nx(self.api, self._archs[i]) for i in sort_inds]
            log_likelihoods = self._log_likelihoods[sort_inds]
            log_offset = log_likelihoods.max()
            shifted_log_likelihoods = log_likelihoods - log_offset
            scaled_likelihoods = shifted_log_likelihoods.exp()
            kernel = [WeisfilerLehman(h=1)]
            surrogate_model = GraphGP(archs, scaled_likelihoods, kernel)
            integrand_model = IntegrandModel(surrogate=surrogate_model)
            with torch.enable_grad():
                surrogate_model.fit(wl_subtree_candidates=(1,))
            with torch.no_grad():
                kernel_int_options = options.get(
                    'kernel_integration',
                    self._numerics.get('kernel_integration', {})
                )
                if kernel_int_options == 'load':
                    kernel_sums = self._kernel_integrals
                else:
                    kernel_sums = integrand_model.compute_kernel_sums(
                        **kernel_int_options,
                        api=self.api,
                        dataset=self.dataset,
                        **self._numerics.get('nas_options', {})
                    )
            quadrature_weights = (
                kernel_sums.view(1, -1) @ surrogate_model.K_i
            ).view(-1)
            while (quadrature_weights < 0).any():
                if 'doubly' in options['method']:
                    # Doubly greedy; drops index immediately if it
                    # reduces the number of negative weights.
                    num_negative_weight = (quadrature_weights < 0).sum()
                    start_length = len(quadrature_weights)
                    logger.info(f'{num_negative_weight.item()}/{start_length} Quadrature Weights Negative')
                    decrement = 0
                    for i in range(len(archs)):
                        i -= decrement
                        indices = [j for j in range(len(archs)) if j != i]
                        archs_ = [archs[j] for j in indices]
                        log_likelihoods_ = log_likelihoods[indices]
                        kernel_sums_ = kernel_sums[indices]
                        log_offset = log_likelihoods_.max()
                        shifted_log_likelihoods = log_likelihoods_ - log_offset
                        scaled_likelihoods = shifted_log_likelihoods.exp()
                        surrogate_model = GraphGP(archs_, scaled_likelihoods, kernel)
                        with torch.enable_grad():
                            surrogate_model.fit(wl_subtree_candidates=(1,))
                        quadrature_weights_ = (
                            kernel_sums_.view(1, -1) @ surrogate_model.K_i
                        ).view(-1)
                        num_negative_weight_ = (quadrature_weights_ < 0).sum()
                        if num_negative_weight_ < num_negative_weight:
                            logger.info(f'Drop {i + decrement}')
                            archs = archs_
                            log_likelihoods = log_likelihoods_
                            kernel_sums = kernel_sums_
                            quadrature_weights = quadrature_weights_
                            num_negative_weight = num_negative_weight_
                            decrement += 1
                    if len(quadrature_weights) == start_length:
                        # Drop all negative
                        logger.info(f'Drop all negative')
                        mask = quadrature_weights > 0
                        archs = [
                            archs[i] for i in range(len(archs)) if mask[i]
                        ]
                        log_likelihoods = log_likelihoods[mask]
                        kernel_sums = kernel_sums[mask]
                        log_offset = log_likelihoods.max()
                        shifted_log_likelihoods = log_likelihoods - log_offset
                        scaled_likelihoods = shifted_log_likelihoods.exp()
                        surrogate_model = GraphGP(archs, scaled_likelihoods, kernel)
                        with torch.enable_grad():
                            surrogate_model.fit(wl_subtree_candidates=(1,))
                        quadrature_weights = (
                            kernel_sums.view(1, -1) @ surrogate_model.K_i
                        ).view(-1)
                else:
                    quadrature_weights_s = []
                    negative_weight_totals = []
                    for i in range(len(archs)):
                        indices = [j for j in range(len(archs)) if j != i]
                        archs_ = [archs[j] for j in indices]
                        log_likelihoods_ = log_likelihoods[indices]
                        kernel_sums_ = kernel_sums[indices]
                        log_offset = log_likelihoods_.max()
                        shifted_log_likelihoods = log_likelihoods_ - log_offset
                        scaled_likelihoods = shifted_log_likelihoods.exp()
                        surrogate_model = GraphGP(archs_, scaled_likelihoods, kernel)
                        with torch.enable_grad():
                            surrogate_model.fit(wl_subtree_candidates=(1,))
                        quadrature_weights_ = (
                            kernel_sums_.view(1, -1) @ surrogate_model.K_i
                        ).view(-1)
                        quadrature_weights_s.append(quadrature_weights_)
                        negative_weight_totals.append(
                            quadrature_weights_[quadrature_weights_ < 0].sum()
                        )
                    max_index = np.argmax(negative_weight_totals)
                    logger.info(f'Drop {max_index}')
                    indices = [i for i in range(len(archs)) if i != max_index]
                    archs = [archs[i] for i in indices]
                    log_likelihoods = log_likelihoods[indices]
                    kernel_sums = kernel_sums[indices]
                    quadrature_weights = quadrature_weights_s[max_index]
            self._archs = archs
            self._log_likelihoods = log_likelihoods
            log_offset = log_likelihoods.max()
            shifted_log_likelihoods = log_likelihoods - log_offset
            scaled_likelihoods = shifted_log_likelihoods.exp()
            surrogate_model = GraphGP(archs, scaled_likelihoods, kernel)
            with torch.enable_grad():
                surrogate_model.fit(wl_subtree_candidates=(1,))
            self._integrand_model = IntegrandModel(surrogate=surrogate_model)
            log_correction = (
                self._log_model_evidence
                - self._integrand_model.posterior(quadrature_weights).loc.log()
                - log_offset
            )
            self._quadrature_weights = (
                log_correction.exp() * quadrature_weights
            )
            self._ensemble_weights = self._quadrature_weights * (
                self._log_likelihoods - self._log_model_evidence
            ).exp()
            level = options.get('level')
            level = level if level is not None else len(self._ensemble_weights)
            self._ensemble_indices = torch.argsort(self._ensemble_weights)[
                -level:
            ]
            return self._ensemble_indices
        elif 'optimise_weights_pretruncated' in options['method']:
            abs_weights = self._ensemble_weights#.abs()
            level = options.get('level')
            level = abs_weights.size(0) if level is None else level
            indices = torch.argsort(abs_weights)[-level:]
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            weights = optimise_weights(
                self._archs,
                self._log_likelihoods,
                self.api,
                self.dataset,
                sum_space=self._numerics.get('sum_space', 'probability'),
                nas_options=self._numerics.get('nas_options', {}),
                data_dir=self.data_dir,
            )
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif 'optimise_weights' == options['method']:
            weights = optimise_weights(
                self._archs,
                self._log_likelihoods,
                self.api,
                self.dataset,
                sum_space=self._numerics.get('sum_space', 'probability'),
                nas_options=self._numerics.get('nas_options', {}),
                data_dir=self.data_dir,
            )
            level = options.get('level')
            level = weights.size(0) if level is None else level
            indices = torch.argsort(weights)[-level:]
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = weights[indices]
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif 'optimise_weights_iterative' == options['method']:
            weights, indices = optimise_weights_iterative(
                self._archs,
                self._log_likelihoods,
                self.api,
                self.dataset,
                sum_space=self._numerics.get('sum_space', 'probability'),
                nas_options=self._numerics.get('nas_options', {}),
                data_dir=self.data_dir,
                # weights=torch.softmax(
                #     self._kernel_integrals @ self._integrand_model.surrogate.K_i
                #     * torch.exp(
                #         self._log_likelihoods - self._log_model_evidence
                #     ), dim=0
                # ),
                weights=self._ensemble_weights,
                level=options.get('level')
            )
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = weights
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        elif 'optimise_weights_kernel_reweighting' == options['method']:
            weights = optimise_weights(
                self._archs,
                self._log_likelihoods,
                self.api,
                self.dataset,
                sum_space=self._numerics.get('sum_space', 'probability'),
                nas_options=self._numerics.get('nas_options', {}),
                data_dir=self.data_dir,
            )
            level = options.get('level')
            level = weights.size(0) if level is None else level
            weights_argsort = torch.argsort(weights)
            indices = weights_argsort[-level:]
            excluded_inds = weights_argsort[:-level]
            reweighting = self._integrand_model.surrogate.K[indices][:, excluded_inds]
            reweighting /= reweighting.sum(dim=0, keepdim=True)
            self._archs = [self._archs[i] for i in indices]
            self._log_likelihoods = self._log_likelihoods[indices]
            self._ensemble_weights = weights[indices] + reweighting @ weights[excluded_inds]
            self._ensemble_indices = torch.arange(self._ensemble_weights.size(0))
            return self._ensemble_indices
        else:
            raise NotImplementedError


    @property
    def archs(self):
        return [self._archs[i] for i in self.ensemble_indices.tolist()]

    @property
    def log_likelihoods(self):
        return self._log_likelihoods[self.ensemble_indices]

    @property
    def ensemble_weights(self):
        try:
            return (
                self._ensemble_weights[self.ensemble_indices]
                / self._ensemble_weights[self.ensemble_indices].sum()
            )
        except KeyError:
            logger.warning(
                'Truncation not implemented for MMLT, using all architectures.'
            )
            return self._ensemble_weights

    def learn(
        self,
        numerics: Mapping = None,
        load_design_set: int = None,
        log_dir: Union[Path, str] = Path('./'),
        sacred_run: Run = None,
        debug_options: Mapping = None
    ) -> None:
        """Infer the model parameters.
        
        :param numerics: Keyword arguments for whichever method is to be
            used.
        :param load_design_set: Sacred Run ID from which to load the
            design set. Expected that this is stored as a list of API
            indices in a file called archs.txt.
        :param log_dir: Logging directory.
        :param sacred_run: Sacred run object.
        :param debug_options: Options for debugging.
        """
        if numerics is None:
            numerics = {'method': 'bayesian_quadrature'}
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        if numerics['method'] == 'bayesian_quadrature':
            learn_output = self._bayesian_quadrature(
                **numerics,
                log_dir=log_dir,
                sacred_run=sacred_run,
                debug_options=debug_options
            )
        elif numerics['method'] == 'deep_ensemble':
            learn_output = self._deep_ensemble(
                **numerics,
                log_dir=log_dir,
                sacred_run=sacred_run,
                debug_options=debug_options
            )
        elif numerics['method'] == 'neural_ensemble_search_regularised_evolution':
            try:
                learn_output = self._neural_ensemble_search_regularised_evolution(
                    **numerics,
                    load_design_set=load_design_set,
                    log_dir=log_dir,
                    sacred_run=sacred_run,
                    debug_options=debug_options
                )
            except TypeError:
                learn_output = self._neural_ensemble_search_regularised_evolution(
                    **numerics,
                    log_dir=log_dir,
                    sacred_run=sacred_run,
                    debug_options=debug_options
                )
        elif numerics['method'] == 'neural_ensemble_search_bayesian_sampling':
            try:
                learn_output = self._neural_ensemble_search_bayesian_sampling(
                    **numerics,
                    load_design_set=load_design_set,
                    log_dir=log_dir,
                    sacred_run=sacred_run,
                    debug_options=debug_options
                )
            except TypeError:
                learn_output = self._neural_ensemble_search_bayesian_sampling(
                    **numerics,
                    log_dir=log_dir,
                    sacred_run=sacred_run,
                    debug_options=debug_options
                )
        return learn_output
    
    def evaluate(
        self,
        learn_output = None,
        numerics: Mapping = None,
        log_dir: Union[Path, str] = Path('./logs'),
        sacred_run: Run = None,
        smoke_test: bool = False
    ) -> None:
        """Evaluate the performance of the ensemble.
        
        :param learn_output: Output of learning strategy.
        :param numerics: Keyword arguments for whichever method is to be
            used.
        :param log_dir: Logging directory.
        :param sacred_run: Sacred run object.
        :param smoke_test: Smoke test flag.
        :return: Metrics on test set.
        """
        if smoke_test:
            test_batch_size = 2
        else:
            test_batch_size = numerics.get('test_batch_size', 100)
        _, _, test_dataloader, class_num = get_data_splits(
            'cifar10' if self.dataset == 'cifar10-valid' else self.dataset,
            self.data_dir, batch_size=test_batch_size
        )
        num_test_data = len(test_dataloader.dataset)
        if self.dataset != 'cifar10-valid':
            num_test_data /= 2
        prob_bin_width = 0.1  # TODO: Specify in YAML config.
        bin_edges_left = np.arange(0, 1, prob_bin_width)
        bin_centres = bin_edges_left + 0.5 * prob_bin_width
        if numerics['method'] == 'bayesian_quadrature':
            if learn_output is not None:
                ensemble = learn_output[0]
                log_likelihoods = learn_output[1]
                if numerics.get('surrogate', {}).get('warping', None) is None:
                    ensemble_weights = learn_output[2]
                else:
                    ensemble_weights = learn_output[3]
            else:
                ensemble = self.archs
                log_likelihoods = self.log_likelihoods
                ensemble_weights = self.ensemble_weights
            best_observed_index = log_likelihoods.argmax().item()
            best_arch = [ensemble[best_observed_index]]
        elif numerics['method'] == 'deep_ensemble':
            ensemble = learn_output[0]
            ensemble_weights = None
            best_arch = ensemble
        elif numerics['method'] == 'neural_ensemble_search_regularised_evolution':
            ensemble = learn_output[0]
            ensemble_weights = None
            best_arch = [ensemble[0]]
        elif numerics['method'] == 'neural_ensemble_search_bayesian_sampling':
            ensemble = learn_output[0]
            log_likelihoods = learn_output[1]
            ensemble_weights = None
            best_observed_index = log_likelihoods.argmax().item()
            best_arch = [ensemble[best_observed_index]]
        else:
            raise NotImplementedError
        (
            test_log_likelihood,
            test_accuracy,
            calibration_error,
            bin_accuracies,
            bin_histogram,
            even_test_log_likelihood,
            even_test_accuracy,
            even_calibration_error,
            even_bin_accuracies,
            even_bin_histogram
        ) = evaluate_ensemble(
            test_dataloader,
            num_test_data,
            class_num,
            ensemble,
            self.api,
            self.dataset,
            split='test',
            seed=numerics.get('nas', {}).get('seed', 777),
            train_epochs=numerics.get('nas', {}).get('train_epochs', "200"),
            weights=ensemble_weights,
            param_ensemble_size=numerics.get(
                'architecture_likelihood', {}
            ).get('ensemble_size', 1),
            param_weights=numerics.get(
                'architecture_likelihood', {}
            ).get('weights', 'even'),
            sum_space=numerics.get('sum_space', 'probability'),
            smoke_test=smoke_test
        ) 
        (
            best_test_log_likelihood,
            best_test_accuracy,
            best_calibration_error,
            best_bin_accuracies,
            best_bin_histogram,
            *_
        ) = evaluate_ensemble(
            test_dataloader,
            num_test_data,
            class_num,
            best_arch,
            self.api,
            self.dataset,
            split='test',
            seed=numerics.get('nas', {}).get('seed', 777),
            train_epochs=numerics.get('nas', {}).get('train_epochs', "200"),
            weights=None,
            param_ensemble_size=1,
            param_weights='even',
            sum_space=numerics.get('sum_space', 'probability'),
            smoke_test=smoke_test
        ) 
        # Log metrics on the test set
        logger.info(f'Test Log Likelihood: {test_log_likelihood.item()}')
        logger.info(f'Test Accuracy: {test_accuracy.item()}')
        logger.info(
            f'Test Expected Calibration Error: {calibration_error.item()}'
        )
        logger.info(
            f'Even Ensemble Test Log Likelihood: {even_test_log_likelihood.item()}'
        )
        logger.info(
            f'Even Ensemble Test Accuracy: {even_test_accuracy.item()}'
        )
        logger.info(
            f'Even Ensemble Test Expected Calibration Error: {even_calibration_error.item()}'
        )
        logger.info(
            f'Best Valid Test Log Likelihood: {best_test_log_likelihood.item()}'
        )
        logger.info(f'Best Valid Test Accuracy: {best_test_accuracy.item()}')
        logger.info(
            f'Best Valid Test Expected Calibration Error: {best_calibration_error.item()}'
        )
        if sacred_run is not None:
            sacred_run.log_scalar(
                metric_name='weighted-ensemble-test-log-likelihood',
                value=test_log_likelihood.item()
            )
            sacred_run.log_scalar(
                metric_name='weighted-ensemble-test-accuracy',
                value=test_accuracy.item()
            )
            sacred_run.log_scalar(
                metric_name='weighted-ensemble-test-expected-calibration-error',
                value=calibration_error.item()
            )
            sacred_run.log_scalar(
                metric_name='even-ensemble-test-log-likelihood',
                value=even_test_log_likelihood.item()
            )
            sacred_run.log_scalar(
                metric_name='even-ensemble-test-accuracy',
                value=even_test_accuracy.item()
            )
            sacred_run.log_scalar(
                metric_name='even-ensemble-test-expected-calibration-error',
                value=even_calibration_error.item()
            )
            sacred_run.log_scalar(
                metric_name='best-valid-test-log-likelihood',
                value=best_test_log_likelihood.item()
            )
            sacred_run.log_scalar(
                metric_name='best-valid-test-accuracy',
                value=best_test_accuracy.item()
            )
            sacred_run.log_scalar(
                metric_name='best-valid-test-expected-calibration-error',
                value=best_calibration_error.item()
            )
        plot_calibration(
            bin_centres, bin_accuracies, bin_histogram, log_dir, 'weighted-ensemble'
        )
        plot_calibration(
            bin_centres,
            even_bin_accuracies,
            even_bin_histogram,
            log_dir,
            'even-ensemble'
        )
        plot_calibration(
            bin_centres,
            best_bin_accuracies,
            best_bin_histogram,
            log_dir,
            'best-valid'
        )
        plot_calibration_comparison(
            bin_centres,
            bin_accuracies,
            even_bin_accuracies,
            best_bin_accuracies,
            bin_histogram,
            even_bin_histogram,
            best_bin_histogram,
            log_dir
        )
        # Save quantities of interest
        np.savez(
            str(log_dir / 'weighted-calibration.npz'),
            bin_accuracies=bin_accuracies,
            bin_histogram=bin_histogram
        )
        np.savez(
            str(log_dir / 'even-calibration.npz'),
            bin_accuracies=even_bin_accuracies,
            bin_histogram=even_bin_histogram
        )
        np.savez(
            str(log_dir / 'best-valid-calibration.npz'),
            bin_accuracies=best_bin_accuracies,
            bin_histogram=best_bin_histogram
        )

    def posterior_predictive(
        self,
        numerics: Mapping,
        test_inputs: torch.Tensor,
        num_classes: int,
        sum_space: str = 'probability',
        seed: int = 777,
        train_epochs: int = 200,
        kernel_integration: Mapping = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute a prediction, marginalising over model parameters.
        
        :param test_inputs: Test inputs, shape [B, C, W, H], where B is 
            the batch dimension, C is the number of channels, W is the
            width and H is the height.
        :param num_classes: The number of clases.
        :param sum_space: Whether to do the weighted sum in the
            'probability' space or the 'logit' space.
        :param seed: The NAS seed. For loading the correct network
            weights from NATS-Bench.
        :param train_epochs: The number of training epochs. For loading
            the correct network weights from NATS-Bench.
        :return: Probabilities assigned to each class for every point in
            the test set. Shape [B, C], where B is the number of test
            points and C is the number of classes. Do this using
            self.ensemble weights, and equal weights.
        """
        surrogate_warping = numerics.get('surrogate', {}).get('warping', None)
        normalise_y = numerics.get('surrogate', {}).get('normalise_y', False)
        # Make predictions with each architecture
        if surrogate_warping is None:
            weighted_posteriors = torch.zeros(
                (test_inputs.size(0), num_classes)
            )
        if normalise_y:
            normalisation_correction = torch.zeros(
                (test_inputs.size(0), num_classes)
            )
        even_posteriors = torch.zeros((test_inputs.size(0), num_classes))
        even_w = 1 / len(self.log_likelihoods)
        if surrogate_warping is not None:
            assert sum_space == 'probability', 'logits are not non-negative!'
            all_posteriors = []
        for arch_ndx, a in enumerate(self.archs):
            logger.info(f'Computing Posterior for Architecture {arch_ndx}')
            # Load trained architecture into memory.
            try:
                try:
                    index = self.api.query_index_by_arch(a.name)
                except AttributeError:
                    index = a
                config = self.api.get_net_config(index, self.dataset)
                network = get_cell_based_tiny_net(config)
                self.api.reload(None, index)
                params_dict = self.api.get_net_param(
                    index, self.dataset, seed=seed, hp=train_epochs
                )
                if seed is None:
                    params_list = params_dict.values()
                    if numerics['method'] == 'deep_ensemble':
                        # Load additional params and append to params_iterable
                        num_weight_weights = numerics['num_weights']
                        weight_weights = torch.ones(num_weight_weights) / num_weight_weights
                        params_list += load_deep_ensembles(num_weight_weights - len(params_list))
                    else:
                        raise NotImplementedError
                    params_iterable = iter(params_list)
                else:
                    params_iterable = iter([params_dict])
                    weight_weights = torch.ones(1)
                a_posteriors = torch.zeros((test_inputs.size(0), num_classes))
                for w, params in zip(weight_weights, params_iterable):
                    network.load_state_dict(params)
                    network.cuda()
                    network.eval()
                    # Get model output for the test set
                    a_posteriors += w * network(test_inputs.cuda())[1].cpu()
                self.api.clear_params(index)
            except Exception as e:
                warnings.warn(f'Architecture {arch_ndx} not included.')
                continue
            a_posteriors = (
                torch.softmax(a_posteriors, 1)
                if sum_space == 'probability' else a_posteriors
            )
            if surrogate_warping is None:
                w = self.ensemble_weights[arch_ndx]
                weighted_posteriors = weighted_posteriors + w * a_posteriors
                if normalise_y:
                    normalisation_correction = (
                        normalisation_correction
                        + self.log_likelihoods[arch_ndx].exp() * a_posteriors
                    )
            else:
                all_posteriors.append(a_posteriors.unsqueeze(1))
            even_posteriors = even_posteriors + even_w * a_posteriors
        if surrogate_warping is None:
            if torch.isnan(self._log_model_evidence):
                model_evidence = (
                    self._quadrature_weights 
                    @ self._integrand_model.surrogate.y
                    + self._integrand_model.surrogate.y_mean
                )
            else:
                model_evidence = self._log_model_evidence.exp()
            if normalise_y:
                normalisation_correction = (
                    normalisation_correction
                    / len(self.log_likelihoods)
                    / model_evidence
                )
                weighted_posteriors = (
                    weighted_posteriors
                    + normalisation_correction
                    * (1 - self._quadrature_weights.sum())
                )
        if sum_space == 'logit':
            weighted_posteriors = torch.softmax(weighted_posteriors, 1)
            even_posteriors = torch.softmax(even_posteriors, 1)
        if surrogate_warping is not None:
            kernel_integration = (
                {} if kernel_integration is None else kernel_integration
            )
            weighted_posteriors = self._integrand_model.posterior_predictive(
                self._quadrature_weights,
                torch.cat(all_posteriors, 1),  # [B, N (num_archs), C]
                log_model_evidence=self._log_model_evidence,
                **kernel_integration
            )
            if not (weighted_posteriors.sum(1) == 1).all():
                logger.warning(
                    'Weighted posteriors not normalised; forcing...'
                )
                weighted_posteriors = torch.softmax(weighted_posteriors, 1)
        return weighted_posteriors, even_posteriors
    
    def _bayesian_quadrature(
        self,
        surrogate: Mapping = None,
        initialisation: Mapping = None,
        evaluation_budget: int = 4,
        kernel_integration: Mapping = None,
        acquisition: Mapping = None,
        nas_options: Mapping = None,
        log_dir: Path = Path('./logs'),
        sacred_run: Run = None,
        debug_options: Mapping = None,
        **kwargs
    ) -> None:
        """Perform Bayesian Quadrature to marginalise over
        architectures.

        :param surrogate: Surrogate options.
        :param initialisation: Initialisation options.
        :param kernel_integration: Kernel integration options.
        :param acquisition: Acquisition function options.
        :param log_dir: Logging directory.
        :param sacred_run: Sacred run object.
        :param debug_options: Options related to debugging.
            smoke_test: whether to run with fastest options.
            test_set: "random", "grid", or None. The test set size will
                always be 625. If "grid" then we select every 25th arch,
                ranked by (the average over seed=[777,888],
                train_epochs=200) validation loss.
        :param kwargs: Swallows redundant arguments.
        """
        (
            ensemble,
            log_likelihoods,
            ensemble_weights,
            integrand_model,
            quadrature_weights,
            log_model_evidence
        ) = bayesian_quadrature(
            self.api,
            self.dataset,
            surrogate=surrogate,
            initialisation=initialisation,
            evaluation_budget=evaluation_budget,
            kernel_integration=kernel_integration,
            acquisition=acquisition,
            nas_options=nas_options,
            data_dir=self.data_dir,
            log_dir=log_dir,
            sacred_run=sacred_run,
            debug_options=debug_options,
            **kwargs
        )
        self._archs = ensemble
        self._log_likelihoods = log_likelihoods
        self._ensemble_weights = ensemble_weights
        self._integrand_model = integrand_model
        self._quadrature_weights = quadrature_weights
        self._log_model_evidence = log_model_evidence
        return (
            ensemble,
            log_likelihoods,
            ensemble_weights,
            integrand_model,
            quadrature_weights,
            log_model_evidence
        )

    def _deep_ensemble(
        self,
        evaluation_budget: int,
        ensemble_size: int,
        debug_options: Mapping = None,
        **kwargs
    ) -> Sequence[int]:
        """Construct deep ensemble of architecture with best validation
        loss.

        :param evaluation_budget: The evaluation budget.
        :param ensemble_size: The ensemble size.
        :param debug_options: Options for debugging.
        """
        ensemble = learn_deep_ensembles(
            evaluation_budget,
            ensemble_size,
            self.api,
            self.dataset,
            self.data_dir,
            smoke_test=debug_options['smoke_test']
        )
        self._archs = ensemble
        return ensemble
    
    def _neural_ensemble_search_regularised_evolution(
        self,
        nas_options: Mapping = None,
        evaluation_budget: int = 150,
        population_size: int = 50,
        num_parent_candidates: int = 10,
        ensemble_size: int = 3,
        load_design_set: int = None,
        log_dir: Path = Path('./'),
        debug_options: Mapping = None,
        **kwargs
    ) -> Tuple[Sequence[Graph], torch.Tensor]:
        """Run Neural Ensemble Search with Regularised Evolution.
        
        :param nas_options: The nas_options.
        :param evaluation_budget: The evaluation budget.
        :param population_size: The population size to evolve.
        :param num_parent_candidates: The number of parent candidates
            for evolution.
        :param ensemble_size: The ensemble size.
        :param load_design_set: The Sacred Run ID from which to load
            the design set.
        :param log_dir: The logging directory.
        :param debug_options: Debug options.
        :return: The architectures, log likelihoods, and equal weights.
        """
        debug_options = {} if debug_options is None else debug_options
        ensemble, log_likelihoods = neural_ensemble_search_regularised_evolution(
            self.api,
            self.dataset,
            nas_options=nas_options,
            data_dir=self.data_dir,
            evaluation_budget=evaluation_budget,
            population_size=population_size,
            num_parent_candidates=num_parent_candidates,
            ensemble_size=ensemble_size,
            load_design_set=load_design_set,
            log_dir=log_dir,
            smoke_test=debug_options.get('smoke_test', False)
        )
        self._archs = ensemble
        self._log_likelihoods = log_likelihoods
        self._ensemble_weights = torch.ones_like(log_likelihoods) / log_likelihoods.size(0)
        return ensemble, log_likelihoods, self._ensemble_weights
    
    def _neural_ensemble_search_bayesian_sampling(
        self,
        load_design_set: int,
        ensemble_size: int = 3,
        iterations: int = 10,
        log_dir: Path = Path('./'),
        debug_options: Mapping = None,
        **kwargs
    ):
        """Neural Ensemble Search by Bayesian Sampling.

        This implementation assumes the candidate architecture set is
        given.
        
        :param ensemble_size: The target ensemble size.
        :param iterations: The number of ensembles to samples before
            selecting the best.
        :param load_design_set: The ID from which to load the candidate
            architecture set.
        :param log_dir: The logging directory.
        :param debug_options: Debug options.
        :return: The architectures, log likelihoods and weights.
        """
        debug_options = {} if debug_options is None else debug_options
        ensemble, log_likelihoods = neural_ensemble_search_bayesian_sampling(
            self.api,
            self.dataset,
            self._numerics.get('nas_options', {}),
            load_design_set,
            ensemble_size=ensemble_size,
            iterations=iterations,
            log_dir=log_dir,
            smoke_test=debug_options.get('smoke_test', False)
        )
        self._archs = ensemble
        self._log_likelihoods = log_likelihoods
        self._ensemble_weights = torch.ones_like(log_likelihoods) / log_likelihoods.size(0)
        return ensemble, log_likelihoods, self._ensemble_weights
