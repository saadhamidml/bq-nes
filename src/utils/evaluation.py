from re import A
from typing import Type, Union, Tuple, Sequence, Mapping, Optional
from pathlib import Path
from copy import deepcopy
import logging
import json
import math
import random
import numpy as np
from networkx import Graph
from sacred.run import Run
import torch
from PIL import Image
from xautodl.datasets import get_datasets
from xautodl.config_utils import load_config
from xautodl.datasets import get_datasets
from nats_bench import NATStopology

from bayes_quad.gp import GraphGP
from bayes_quad.quadrature import IntegrandModel
from model.deep_ensemble import load_deep_ensemble
from utils.nas import load_architecture
from utils.plotting import plot_surrogate_metrics


logger = logging.getLogger(__name__)


TEST_BATCH_SIZE = {
    'cifar10-valid': 10000,
    'cifar100': 5000,
    'ImageNet16-120': 3000
}


def get_test_metrics(
    targets: torch.Tensor,
    posteriors: torch.Tensor,
    bin_edges_left: np.ndarray,
    batch_size: int = None,
    return_bin_histogram: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
]:
    """Compute test log likelihood, overall accuracy and
    accuracy per probability bin.
    
    :param targets: The targets. Shape [M].
    :param posteriors: Probabilities assigned to each class for
        every point in the test set. Shape [M, C], where M is
        the number of test points and C is the number of
        classes.
    :param bin_edges_left: The left bin edges for the
        probability interval.
    :param return_bin_histogram: Whether to return a histogram
        of number of predictive probabilities falling in each
        bin.
    """
    batch_size = targets.size(0) if batch_size is None else batch_size
    pred_probs, preds = torch.max(posteriors, dim=1)
    correct_preds = preds == targets
    test_accuracy = correct_preds.to(dtype=torch.float).sum() / batch_size
    test_log_likelihood = posteriors[
        torch.arange(posteriors.size(0)), targets
    ].log().sum()
    
    bin_inds = np.digitize(pred_probs.detach().numpy(), bin_edges_left) - 1
    bin_accuracies = []
    for i in range(len(bin_edges_left)):
        mask = bin_inds == i
        i_accuracy = correct_preds[mask].to(dtype=torch.float).mean()
        bin_accuracies.append(i_accuracy.item())
    bin_accuracies = np.array(bin_accuracies)
    if return_bin_histogram:
        return (
            test_log_likelihood,
            test_accuracy,
            bin_accuracies,
            np.bincount(bin_inds, minlength=len(bin_edges_left))
        )
    else:
        return test_log_likelihood, test_accuracy, bin_accuracies


def predict_parameter_ensemble(
    inputs: torch.Tensor,
    num_classes: int,
    network: torch.nn.Module,
    params_dict: Optional[Mapping] = None,
    param_weights: str = 'even',
    sum_space: str = 'probability',
    logit_paths: Sequence[Path] = None
) -> torch.Tensor:
    """Prediction for an ensemble consisting of different parameter
    settings for the same architecture.
    
    :param inputs: The predictands.
    :param num_classes: The number of classes.
    :param network: The network.
    :param params_dict: Dictionary of parameter settings.
    :param_weights: Method of weighting the parameter settings.
    :param sum_space: Whether to sum in the probability or logit space.
    """
    if logit_paths is None:
        raise TypeError
    params_list = params_dict.values()
    params_iterable = iter(params_list)
    if param_weights == 'even':
        ensemble_size = len(params_list)
        param_weights = torch.ones(ensemble_size) / ensemble_size
    else:
        raise NotImplementedError
        lls = []
        seeds = [777, 888]
        num_batches = None
        for s in seeds:
            ll = -self.api.get_more_info(
                a, self.dataset, hp=train_epochs, is_random=s
            )['train-loss'] * num_batches
            if not np.isnan(ll):
                lls.append(ll)
        param_weights = np.exp(
            np.array(lls) - scipy.special.logsumexp(lls)
        )
    posteriors = torch.zeros((inputs.size(0), num_classes))
    for w, p, lp in zip(param_weights, params_iterable, logit_paths):
        try:
            lp.parent.mkdir(parents=True, exist_ok=True)
            logits = torch.load(str(lp))
        except FileNotFoundError as e:
            network.load_state_dict(p)
            network.cuda()
            network.eval()
            logits = network(inputs.cuda())[1].cpu()
            torch.save(logits, str(lp))
        softmaxxed = torch.softmax(logits, 1) if sum_space == 'probability' else logits
        posteriors += w * softmaxxed
    return posteriors


def evaluate_ensemble_for_batch(
    inputs: torch.Tensor,
    num_classes: int,
    ensemble: Sequence[Union[Graph, int]],
    api: NATStopology,
    dataset: str,
    split: str = None,
    seed: int = None,
    train_epochs: str = "200",
    weights: torch.Tensor = None,
    param_ensemble_size: int = 1,
    param_weights: str = 'even',
    sum_space: str = 'probability',
    bin_edges_left: np.ndarray = None,
    batch_size: int = None,
    batch_index: int = 0,
    targets: torch.Tensor = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Evaluate ensemble for a batch.
    
    :param inputs: The predictands.
    :param num_classes: The number of classes.
    :param ensemble: The ensemble members.
    :param api: The api.
    :param seed: The NAS seed.
    :param train_epochs: The number of training epochs.
    :param weights: The ensemble weights.
    :param param_ensemble_size: The number of parameter settings to use
        for the marginalisation of architecture parameters.
    :param param_weights: If seed is not None -- how to weight the
        different seeds. Options are 'even' or 'likelihood'.
    :param sum_space: Whether to sum in the probability or logit space.
    :param bin_edges_left: Left bin edges. Needed only if targets
        provided.
    :param batch_size: The batch size.
    :param batch_index: The batch index.
    :param targets: The targets.
    :return: The posteriors, and the metrics for the batch.
    """
    if split is None:
        raise TypeError
    posteriors = torch.zeros((inputs.size(0), num_classes))  # Weighted
    e_posteriors = torch.clone(posteriors)  # Even
    a_posteriors = []  # All (for Warped BQ)
    if isinstance(weights, IntegrandModel) or weights is None:
        integrand_model = weights
        weights = torch.ones(len(ensemble)) / len(ensemble)
    else:
        integrand_model = None
    for arch_ndx, (w, a) in enumerate(zip(weights, ensemble)):
        logger.debug(f'Computing Posterior for Architecture {arch_ndx}')
        try:
            api_index = api.query_index_by_arch(a.name)
        except AttributeError:
            api_index = a
        logit_filename = f'{batch_size}_{batch_index}.pt'
        logit_basepath = Path(f'../data/logits/{api_index}/{dataset}/{split}')
        if seed is None:
            logit_paths = [
                logit_basepath / f'777/{logit_filename}',
                logit_basepath / f'888/{logit_filename}'
            ]
            if param_ensemble_size > 2:
                raise NotImplementedError
        else:
            logit_paths = [logit_basepath / f'{seed}/{logit_filename}']
        if all([lp.exists() for lp in logit_paths]):
            network = None
            params_dict = {lp: None for lp in logit_paths}
        else:
            network, params_dict = load_architecture(
                a, api, dataset, seed=seed, hp=train_epochs
            )
        params_dict.update(load_deep_ensemble(
            param_ensemble_size, api_index, dataset
        ))
        with torch.no_grad():
            output = predict_parameter_ensemble(
                inputs,
                num_classes,
                network,
                params_dict,
                param_weights,
                sum_space,
                logit_paths=logit_paths
            )
        e_posteriors += output / weights.size(0)
        if integrand_model is not None:
            a_posteriors.append(output.unsqueeze(1))
        else:
            posteriors += w * output
    if sum_space == 'logit':
        posteriors = torch.softmax(posteriors, 1)
        e_posteriors = torch.softmax(e_posteriors, 1)
    if integrand_model is not None:
        posteriors = integrand_model.posterior_predictive(
            integrand_model._quadrature_weights,
            torch.cat(a_posteriors, 1),
            integrand_model._log_model_evidence
        )
    if targets is not None:
        batch_size = inputs.size(0) if batch_size is None else batch_size
        metrics = get_test_metrics(
            targets, posteriors, bin_edges_left, batch_size, return_bin_histogram=True
        )
        e_metrics = get_test_metrics(
            targets, e_posteriors, bin_edges_left, batch_size, return_bin_histogram=True
        )
    else:
        metrics = None
        e_metrics = None
    return posteriors, metrics, e_posteriors, e_metrics


def evaluate_ensemble(
    dataloader: torch.utils.data.DataLoader,
    num_data: int,
    num_classes: int,
    ensemble: Sequence[Union[Graph, int]],
    api: NATStopology,
    dataset: str,
    split: str = None,
    seed: int = None,
    train_epochs: str = "200",
    weights: torch.Tensor = None,
    param_ensemble_size: int = None,
    param_weights: str = 'even',
    sum_space: str = 'probability',
    smoke_test: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ensemble on data in dataloader.
    
    :param dataloader: The dataloader.
    :param num_data: The number of data points.
    :param num_classes: The number of classes.
    :param ensemble: The members of the ensemble.
    :param api: The api.
    :param dataset: The dataset.
    :param split: The data split. Required for caching.
    :param seed: The NAS seed.
    :param train_epochs: The number of training epochs.
    :param weights: The ensemble weights.
    :param param_ensemble_size: The number of parameter settings to use
        to marginalise over the architecture parameters.
    :param param_weights: If seed is not None -- how to weight the
        different seeds. Options are 'even' or 'likelihood'.
    :param sum_space: Whether to sum in the probability or logit space.
    :return: The metrics on the dataset.
    """
    if split is None:
        raise TypeError
    prob_bin_width = 0.1  # TODO: Specify in YAML config.
    bin_edges_left = np.arange(0, 1, prob_bin_width)
    log_likelihood = torch.tensor(0.)
    accuracy = torch.tensor(0.)
    bin_accuracies = np.zeros_like(bin_edges_left)
    bin_histogram = np.zeros_like(bin_edges_left)
    e_log_likelihood = torch.tensor(0.)
    e_accuracy = torch.tensor(0.)
    e_bin_accuracies = np.zeros_like(bin_edges_left)
    e_bin_histogram = np.zeros_like(bin_edges_left)
    bin_centres = bin_edges_left + 0.5 * prob_bin_width
    batch_size = dataloader.batch_size
    with torch.no_grad():
        # Set seed to zero to ensure same batches loaded every time.
        rand_state = random.getstate()
        np_rand_state = np.random.get_state()
        torch_rand_state = torch.random.get_rng_state()
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)
        for batch_idx, batch in enumerate(dataloader):
            logger.debug(f'Computing Posterior for Batch {batch_idx}')
            # Compute posterior
            _, metrics, _, e_metrics = evaluate_ensemble_for_batch(
                batch[0],
                num_classes,
                ensemble,
                api,
                dataset,
                split=split,
                seed=seed,
                train_epochs=train_epochs,
                weights=weights,
                param_ensemble_size=param_ensemble_size,
                param_weights=param_weights,
                sum_space=sum_space,
                bin_edges_left=bin_edges_left,
                batch_size=batch_size,
                batch_index=batch_idx,
                targets=batch[1]
            )
            log_likelihood += metrics[0]
            accuracy += metrics[1]
            bin_accuracies += np.nan_to_num(metrics[2] * metrics[3])
            bin_histogram += metrics[3]
            e_log_likelihood += e_metrics[0]
            e_accuracy += e_metrics[1]
            e_bin_accuracies += np.nan_to_num(e_metrics[2] * e_metrics[3])
            e_bin_histogram += e_metrics[3]
            if smoke_test and batch_idx > 0:
                break
        # Restore random states
        random.setstate(rand_state)
        np.random.set_state(np_rand_state)
        torch.set_rng_state(torch_rand_state)
    batching_correction = batch_size / num_data
    accuracy *= batching_correction
    bin_accuracies /= bin_histogram
    calibration_error = np.nansum(
        bin_histogram / num_data * np.abs(bin_accuracies - bin_centres)
    )
    e_accuracy *= batching_correction
    e_bin_accuracies /= e_bin_histogram
    e_calibration_error = np.nansum(
        e_bin_histogram / num_data * np.abs(e_bin_accuracies - bin_centres)
    )
    # quick error check
    assert np.allclose(
        accuracy.numpy(), np.nansum(bin_accuracies * bin_histogram) / num_data
    )
    assert np.allclose(
        e_accuracy.numpy(), np.nansum(e_bin_accuracies * e_bin_histogram) / num_data
    )
    return (
        log_likelihood,
        accuracy,
        calibration_error,
        bin_accuracies,
        bin_histogram,
        e_log_likelihood,
        e_accuracy,
        e_calibration_error,
        e_bin_accuracies,
        e_bin_histogram
    )


def get_test_archs(
    sampling: str, dataset: str, log_dir: Path = Path('../logs')
) -> Sequence[int]:
    """Get test set.
    
    :param sampling: Method of sampling. 'random', 'grid', or 'None'.
    :return: List of architectures (as API indices).
    """
    if sampling == 'random':
        raise NotImplementedError()
    elif sampling == 'grid':
        # Load ranked architectures.
        pickle = np.load(
            f'{log_dir}/sandbox/nas/{dataset}/ranked_architectures.npz'
        )
        indices = 25 * np.arange(int(15625 / 25))
        test_archs = pickle['ranking'][indices].tolist()
    else:
        test_archs = None
    return test_archs


def get_surrogate_performance_metrics(
    locations: Sequence[Graph],
    log_likelihoods: torch.Tensor,
    surrogate: GraphGP,
    log_dir: Path = Path('./'),
    split: str = 'test',
    sacred_run: Run = None
) -> Tuple[float, float]:
    """Get RMSE and NLPD on the set for the surrogate.
    
    If the surrogate is warped we report metrics only for the unwarped
    targets.
    
    :param locations: The inputs.
    :param log_likelihoods: The log of the targets.
    :param surrogate: The surrogate.
    :param log_dir: The logging directory.
    :param split: Whether the train set or test set was input.
    :param sacred_run: Sacred Run object.
    :return: The RMSE and NLPD of the test set.
    """
    targets = log_likelihoods.exp()
    normalisation = targets.mean()

    with torch.no_grad():
        mean, cov = surrogate.predict_unwarped(locations)
    mean *= math.exp(surrogate.log_offset)
    variance = cov.diag()

    error = targets - mean

    normalised_rmse = error.div(normalisation).pow(2).mean().sqrt().item()
    nlpd = -torch.mean(
        - 0.5 * error ** 2 / variance
        - 0.5 * variance.log()
        - 0.5 * math.log(2 * math.pi)
    ).item()

    if sacred_run is not None:
        sacred_run.log_scalar(
            metric_name=f'surrogate.{split}_normalised_rmse',
            value=normalised_rmse
        )
        sacred_run.log_scalar(
            metric_name=f'surrogate.{split}_nlpd', value=nlpd
        )
    
    plot_surrogate_metrics(
        error, variance.sqrt(), log_dir=log_dir, split=split
    )

    return normalised_rmse, nlpd
