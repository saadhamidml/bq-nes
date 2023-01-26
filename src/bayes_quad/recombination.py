from copy import deepcopy
import logging
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Type, Union
import random
from nats_bench.api_topology import NATStopology
from networkx import Graph
import numpy as np
import torch
from bayes_quad.utils import compute_pd_inverse
from utils.evaluation import evaluate_ensemble_for_batch

from utils.nas import random_sampling, get_data_splits, VALID_BATCH_SIZES

from .gp import GraphGP


logger = logging.getLogger(__name__)


def noise_until_positive(
    kernel_integrals: torch.Tensor,
    K_matrix: torch.Tensor
):
    """Add noise until the quadrature weights are positive.
    
    :param kernel_integrals: The kernel integrals.
    :param K_matrix: The K matrix.
    :return: The quadrature weights.
    """
    exponent = -5
    K_i, _ = compute_pd_inverse(K_matrix, jitter=10 ** exponent)
    weights = kernel_integrals @ K_i
    while (weights < 0).any():
        exponent += 1
        K_i, _ = compute_pd_inverse(K_matrix, jitter=10.0 ** exponent)
        weights = kernel_integrals @ K_i
    return weights


def nystrom_test_functions(
    api: NATStopology,
    dataset: str,
    train_epochs: str,
    archs: Sequence[Graph],
    surrogate: GraphGP,
    level: int = None,
    return_integrals: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute test functions at input archs.
    
    :param api: NATS-Bench API.
    :param dataset: The dataset.
    :param train_epochs: The number of training epochs. 
    :param archs: N architectures, shape [N].
    :param surrogate. The trained surrogate.
    :param level: The truncation level, M.
    :param return_integrals: Whether to return the integrals of the
        test functions.
    :return: [M + 1, N] matrix of test functions (rows)
        evaluated at each architecture (cols).
    """
    level = len(archs) if level is None else level
    # eigvals sorted in ascending order, but since we invert the largest
    # eigenvalues contribute the smallest weights.
    eigvals, eigvecs = torch.linalg.eigh(surrogate.K)
    evals = eigvals[:level]  # [M]
    evecs = eigvecs[:, :level]  # [N, M]
    # [N, N]
    cross_cov = surrogate.cross_covariance(archs)
    # [M, N]
    nystrom_test = evecs.t() @ cross_cov
    # Including diff_test requires knowing the truncation
    # level beforehand.
    # [1, N]
    diff_test = 1 - (2 * nystrom_test / evals.view(-1, 1)).sum(
        dim=0, keepdim=True
    )
    test_functions = torch.cat((nystrom_test, diff_test), dim=0)  # [M + 1, M]
    if return_integrals:
        # Draw samples.
        samples = []
        samples_remaining = 2048
        while samples_remaining > 0:
            samples = random_sampling(
                samples_remaining,
                api=api,
                dataset=dataset,
                hp=train_epochs,
            )
            samples = [s for s in samples if s is not None]
            samples_remaining -= len(samples)
        mc_cross_cov = surrogate.cross_covariance(samples)  # [N, 2048]
        mc_nystrom_test = evecs.t() @ mc_cross_cov  # [M, 2048]
        mc_diff_test = 1 - (2 * mc_nystrom_test / evals.view(-1, 1)).sum().view(-1)
        test_integrals = torch.cat((mc_nystrom_test.sum(1), mc_diff_test), dim=0)  # [M + 1]
    else:
        test_integrals = None
    return test_functions, test_integrals


def tchernychova_lyons_simplex(
    A: torch.Tensor, b: torch.Tensor, dantzig: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function implements the simplex algorithm outlined in
    Section 1.4.0.1 of Tchernychova's thesis.

    :param A: Matrix of test functions (rows) evaluated at each input
        (columns). Shape [M, N].
    :param b: The integrals of each of the test functions. These must be
        non-negative. Shape [M].
    :param dantzig: Whether to use the Dantzig rule. If True then the
        direction chosen each iteration is the one that most reduces
        the objective function. If False then choose the element with
        highest index that reduces the cost (this assumes that the
        architectures are sorted in ascending order by likelihood).
    :return: The recombined weights [N + M].
    """
    # assert (b > 0).all()
    weights = torch.cat((torch.zeros(A.size(1)), b), dim=0)  # [N + M]
    D_mat = torch.cat((A, torch.eye(b.size(0))), dim=1)  # [M, N + M]
    obj_vec = torch.cat(
        (torch.zeros(A.size(1)), torch.ones_like(b)), dim=0
    )  # [N + M]

    # At most M iterations.
    counter = 0
    while obj_vec @ weights > 0:
        logger.info(f'TL iteration: {counter}')
        mask = weights == 0  # Possible directions.
        num_candidates = mask.sum().to(torch.int)
        if num_candidates == 0:
            logging.warning('No remaining candidate directions.')
        # Columns with nonzero weight, [M, N + M - C].
        D_weighted = D_mat[:, ~mask]  
        D_zero = D_mat[:, mask]  # Columns with zero weight, [M, C].
        candidate_partials = torch.linalg.solve(D_weighted, -D_zero)  # [N + M - C, C]
        candidates = torch.empty(weights.size(0), num_candidates)  # [N + M, C]
        candidates[mask] = torch.eye(num_candidates)
        candidates[~mask] = candidate_partials
        reduced_costs = obj_vec @ candidates  # [C]
        if (reduced_costs > 0).all():
            break
        if dantzig:
            direction = candidates[:, torch.argmin(reduced_costs)]
        else:
            offset = mask[-b.size(0):].sum().to(torch.int)
            if offset == 0 or (reduced_costs[:-offset] > 0).all():
                reduced_costs_ = reduced_costs
            else:
                reduced_costs_ = reduced_costs[:-offset]
            indices = torch.arange(reduced_costs_.size(0))
            direction = candidates[:, indices[reduced_costs_ < 0][-1]]
        if (direction > 0).all():
            logging.warning(
                f'LP problem is unbounded. Terminating after {counter} iterations.'
            )
        # Maximum step size.
        lambda_max = torch.min((weights / -direction)[direction < 0])
        weights = weights + lambda_max * direction
        if (weights != 0).sum() > b.size(0):
            weights[weights < b.min()] = torch.zeros_like(weights[weights < b.min()])
        counter += 1

    weights[-b.size(0):] = torch.zeros_like(weights[-b.size(0):])
    mask = weights > 0
    weights = weights[mask]
    indices = torch.arange(mask.size(0))[mask]

    return weights, indices


def tchernychova_lyons_car(
    X: torch.Tensor, mu: torch.Tensor, DEBUG=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use SVD method to reduce the number of quadrature nodes. Assumes
    already positively weighted.
    
    :param X: Test functions evaluated at each node. Shape [M, N].
    :param mu: The measure. Must be non-negative. Shape [N].
    :return: The reduced weights, shape [M], and the indices of the
        reduced nodes, shape [M].
    """
    # this functions reduce X from N points to n+1
    X = torch.cat([torch.ones(X.size(0)).unsqueeze(0).T, X], dim=1)
    N, n = X.shape
    U, Sigma, V = torch.linalg.svd(X.T)
    U = torch.cat([U, torch.zeros((n, N-n))], dim=1)
    Sigma = torch.cat([Sigma, torch.zeros(N-n)])
    Phi = V[-(N-n):, :].T
    cancelled = torch.tensor([], dtype=int)

    for _ in range(N-n):
        lm = len(mu)
        plis = Phi[:, 0] > 0
        alpha = torch.zeros(lm)
        alpha[plis] = mu[plis] / Phi[plis, 0]
        idx = torch.arange(lm)[plis]
        idx = idx[torch.argmin(alpha[plis])]
        
        if len(cancelled) == 0:
            cancelled = idx.unsqueeze(0)
        else:
            cancelled = torch.cat([cancelled, idx.unsqueeze(0)])
        mu[:] = mu-alpha[idx]*Phi[:, 0]
        mu[idx] = 0.

        if DEBUG and (not torch.allclose(torch.sum(mu), 1.)):
            # print("ERROR")
            print("sum ", torch.sum(mu))

        Phi_tmp = Phi[:, 0]
        Phi = Phi[:,1:]
        #breakpoint()
        Phi = Phi - torch.matmul(
            Phi[idx].unsqueeze(1),
            Phi_tmp.unsqueeze(1).T,
        ).T/Phi_tmp[idx]
        Phi[idx, :] = 0.

    w_star = mu[mu > 0]
    idx_star = torch.arange(N)[mu > 0]
    return w_star, idx_star


def optimise_weights_iterative(
    archs: Sequence[Union[Graph, int]],
    log_likelihoods: torch.Tensor,
    api: NATStopology,
    dataset: str,
    sum_space: str = 'probability',
    nas_options: Mapping = None,
    data_dir: Path = Path('../data'),
    weights: torch.Tensor = None,
    level: int = None,
    smoke_test: bool = False
):
    if level is None:
        raise TypeError
    _, valid_loader, _, num_classes = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset,
        data_dir=data_dir, batch_size=VALID_BATCH_SIZES[dataset]
    )
    indices = np.arange(weights.size(0)).tolist()
    weights = optimise_weights(
        [archs[i] for i in indices],
        log_likelihoods[indices],
        api,
        dataset,
        sum_space=sum_space,
        nas_options=nas_options,
        data_dir=data_dir,
        valid_loader=valid_loader,
        num_classes=num_classes,
        # weights=weights
    )
    while len(weights) > level:
        logger.info(f'{len(weights)}')
        drop_ind = weights.argmin()
        undropped_inds = torch.arange(weights.size(0)).tolist()
        undropped_inds.pop(drop_ind.item())
        weights = weights[undropped_inds]
        indices.pop(drop_ind.item())
        weights = optimise_weights(
            [archs[i] for i in indices],
            log_likelihoods[indices],
            api,
            dataset,
            sum_space=sum_space,
            nas_options=nas_options,
            data_dir=data_dir,
            valid_loader=valid_loader,
            num_classes=num_classes,
            weights=weights
        )
    return weights, indices


def optimise_weights_wce(
    archs: Sequence[Union[Graph, int]],
    log_likelihoods: torch.Tensor,
    K_mat: torch.Tensor,
    kernel_integrals: torch.Tensor,
    weights = None
) -> torch.Tensor:
    if weights is None:
        raw_weights = (log_likelihoods.clone() - log_likelihoods.mean()) / log_likelihoods.std()
    else:
        raw_weights = weights.log().detach()
    raw_weights.requires_grad_(True)
    optimiser = torch.optim.LBFGS([raw_weights], max_iter=5000, line_search_fn='strong_wolfe')
    def closure():
        optimiser.zero_grad()
        weights = torch.softmax(raw_weights, 0)
        loss = weights @ K_mat @ weights - 2 * weights @ kernel_integrals
        loss.backward()
        return loss
    # for i in range(num_iterations):
    optimiser.step(closure)
    raw_weights.requires_grad_(False)
    return torch.softmax(raw_weights, 0)


def optimise_weights(
    archs: Sequence[Union[Graph, int]],
    log_likelihoods: torch.Tensor,
    api: NATStopology,
    dataset: str,
    sum_space: str = 'probability',
    nas_options: Mapping = None,
    data_dir: Path = Path('../data'),
    num_iterations: int = 1000,
    smoke_test: bool = False,
    valid_loader = None,
    num_classes = None,
    weights = None
) -> torch.Tensor:
    """Optimise the quadrature weights to achieve good performance on
    the validation set.
    
    :param archs: The design set of architectures.
    :param log_likelihoods: The corresponding log likelihoods.
    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    """
    if weights is None:
        raw_weights = (log_likelihoods.clone() - log_likelihoods.mean()) / log_likelihoods.std()
    else:
        raw_weights = weights.log().detach()
    raw_weights.requires_grad_(True)
    # optimiser = torch.optim.SGD([raw_weights], lr=1e-1)
    optimiser = torch.optim.LBFGS([raw_weights], max_iter=5000, line_search_fn='strong_wolfe')
    if valid_loader is None or num_classes is None:
        _, valid_loader, _, num_classes = get_data_splits(
            'cifar10' if dataset == 'cifar10-valid' else dataset,
            data_dir=data_dir, batch_size=VALID_BATCH_SIZES[dataset]
        )
    num_data = len(valid_loader.dataset) / 2  # Validation set is always half a DataLoader.
    prob_bin_width = 0.1  # TODO: Specify in YAML config.
    bin_edges_left = np.arange(0, 1, prob_bin_width)
    def closure():
        optimiser.zero_grad()
        rand_state = random.getstate()
        np_rand_state = np.random.get_state()
        torch_rand_state = torch.random.get_rng_state()
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)
        for batch_idx, batch in enumerate(valid_loader):
            logger.debug(f'Computing Posterior for Batch {batch_idx}')
            # Compute posterior
            _, metrics, *_ = evaluate_ensemble_for_batch(
                batch[0],
                num_classes,
                archs,
                api,
                dataset,
                split='valid',
                **nas_options,
                weights=torch.softmax(raw_weights, 0),
                param_ensemble_size=1,
                param_weights='even',
                sum_space=sum_space,
                bin_edges_left=bin_edges_left,
                batch_size=valid_loader.batch_size,
                batch_index=batch_idx,
                targets=batch[1]
            )
            if smoke_test and batch_idx > 0:
                break
        # Restore random states
        random.setstate(rand_state)
        np.random.set_state(np_rand_state)
        torch.set_rng_state(torch_rand_state)
        loss = -metrics[0]
        loss.backward()
        return loss
    # for i in range(num_iterations):
    optimiser.step(closure)
    raw_weights.requires_grad_(False)
    return torch.softmax(raw_weights, 0)
