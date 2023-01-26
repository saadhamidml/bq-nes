from typing import Sequence
from networkx import Graph
import sklearn
import torch

from bayes_quad.gp import GraphGP
from bayes_quad.utils import compute_pd_inverse


def likelihood_determinant_objective(
    log_likelihoods: torch.Tensor,
    K_matrix: torch.Tensor,
    level: int
) -> torch.Tensor:
    """A simple heuristic that greedily maximises an objective that
    trades off performance and diversity.
    
    objective = sum of likelihoods + determinant of K matrix.

    :param log_likelihoods: The log likelihoods of each architecture.
        Shape [N]
    :param K_matrix: The K matrix. Shape [N, N].
    :param level: The truncation level, M.
    :return: The indices of the truncated design set.
    """
    # Start with the highest likelihood architecture.
    argmax = torch.argmax(log_likelihoods).item()
    indices = [argmax]
    unused = torch.arange(log_likelihoods.size(0))
    unused = unused[unused != argmax]

    for _ in range(level - 1):
        l_design = log_likelihoods[indices].exp().sum()
        l_candidates = log_likelihoods[unused].exp()
        likelihood_part = sklearn.preprocessing.StandardScaler().fit_transform(
            (l_candidates + l_design).view(-1, 1).numpy()
        ).reshape(-1)

        K_design = K_matrix[indices, :][:, indices]
        tmp1 = torch.solve(K_matrix[indices, :][:, unused], K_design)[0]
        tmp2 = (K_matrix[unused, :][:, indices] @ tmp1).diag()
        candidate_dets = K_matrix[unused, unused] - tmp2
        det_part = sklearn.preprocessing.StandardScaler().fit_transform(
            (K_design.det() * candidate_dets).view(-1, 1).numpy()
        ).reshape(-1)

        objective = torch.tensor(likelihood_part) - torch.tensor(det_part)
        argmax = torch.argmax(objective).item()
        indices.append(unused[argmax].item())
        unused = unused[unused != unused[argmax]]
    
    return torch.tensor(indices)


def greedy_pairwise_reduction(
    log_likelihoods: torch.Tensor,
    K_matrix: torch.Tensor,
    level: int
) -> torch.Tensor:
    """Iteratively remove the architecture in the closest pair with the
    lower likelihood.
    
    :param log_likelihoods: The log likelihoods of each architecture.
        Shape [N]
    :param K_matrix: The K matrix. Shape [N, N].
    :param level: The truncation level, M.
    :return: The indices of the truncated design set.
    """
    indices = torch.arange(log_likelihoods.size(0))
    # Drop bottom half to start off with.
    sort_inds = torch.argsort(log_likelihoods)[int(log_likelihoods.size(0) / 2):]
    mask = (indices == sort_inds.view(-1, 1)).sum(dim=0).to(torch.bool)
    indices = indices[mask]
    ensemble_size = len(indices)
    while ensemble_size > level:
        K_remaining = K_matrix[indices, :][:, indices] - torch.eye(len(indices))
        argmax = K_remaining.argmax()
        row = int(argmax / ensemble_size)
        col = argmax % ensemble_size
        if log_likelihoods[indices[row]] < log_likelihoods[indices[col]]:
            drop = row
        else:
            drop = col
        indices = indices[indices != indices[drop]]
        ensemble_size = len(indices)
    return indices


def reduce_partial_correlations(
    log_likelihoods: torch.Tensor,
    K_matrix: torch.Tensor,
    level: int
) -> torch.Tensor:
    """Iteratively remove the architecture in the pair with the largest
    negative partial correlation that has the lower likelihood.

    Note that, if there are no more negative partial correlations the
    function will terminate early. There will still be negative partial
    correlations upon termination.

    :param log_likelihoods: The log likelihoods.
    :param K_matrix: The full K matrix.
    :param level: The desired truncation level.
    :return: The truncated indices, and the new inverted K matrix.
    """
    indices = torch.arange(log_likelihoods.size(0))
    sort_inds = torch.argsort(log_likelihoods)[int(2 * log_likelihoods.size(0) / 3):]
    mask = (indices == sort_inds.view(-1, 1)).sum(dim=0).to(torch.bool)
    indices = indices[mask]
    ensemble_size = len(indices)
    K_remaining = K_matrix[indices, :][:, indices]
    K_i, _ = compute_pd_inverse(K_remaining)
    partial_corrs = K_i / (K_i.diag().view(-1, 1) @ K_i.diag().view(1, -1)).sqrt()
    partial_corrs_ = partial_corrs.abs() - partial_corrs.diag() * torch.eye(len(partial_corrs))
    argmax = partial_corrs_.argmax()
    while ensemble_size > level:
        row = int(argmax / ensemble_size)
        col = argmax % ensemble_size
        if log_likelihoods[indices[row]] < log_likelihoods[indices[col]]:
            drop = row
        else:
            drop = col
        indices = indices[indices != indices[drop]]
        ensemble_size = len(indices)
        K_remaining = K_matrix[indices, :][:, indices]
        K_i, _ = compute_pd_inverse(K_remaining)
        partial_corrs = K_i / (K_i.diag().view(-1, 1) @ K_i.diag().view(1, -1)).sqrt()
        partial_corrs_ = partial_corrs.abs() - partial_corrs.diag() * torch.eye(len(partial_corrs))
        argmax = partial_corrs_.argmax()
    return indices


def greedy_negative_elimination(level: int):
    #TODO: Move from arch_ensemble.py to here.
    pass


def uncertainty_sampling(level: int):
    #TODO: Move from arch_ensemble.py to here.
    pass
