import logging
from pathlib import Path
import pytest
import random
import numpy as np
import torch
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

import sys
sys.path.append('../src')

from utils.nas import get_data_splits
from utils.evaluation import get_test_metrics

@pytest.fixture
def get_test_batch_size():
    return {'cifar10-valid': 10000, 'cifar100': 5000, 'ImageNet16-120': 3000}

def get_accuracy(dataloader, network):
    network.cuda()
    bin_edges_left = np.arange(0, 1, 0.1)
    log_likelihood = torch.tensor(0.)
    accuracy = torch.tensor(0.)
    bin_accuracies = np.zeros_like(bin_edges_left)
    bin_histogram = np.zeros_like(bin_edges_left)
    bin_centres = bin_edges_left + 0.05
    with torch.no_grad():
        for batch_ndx, batch in enumerate(dataloader):
            posterior = network(batch[0].cuda())[1].log_softmax(1).exp_().cpu()
            (
                log_likelihood_batch,
                accuracy_batch,
                bin_accuracies_batch,
                bin_histogram_batch
            ) = get_test_metrics(
                batch[1],
                posterior,
                bin_edges_left,
                batch_size=dataloader.batch_size,
                return_bin_histogram=True
            )
            log_likelihood += log_likelihood_batch
            accuracy += accuracy_batch
            bin_accuracies += np.nan_to_num(
                bin_accuracies_batch * bin_histogram_batch
            )
            bin_histogram += bin_histogram_batch
        batching_correction = dataloader.batch_size / len(dataloader.dataset)
        accuracy *= 100 * batching_correction
        bin_accuracies *= 100 / bin_histogram
        calibration_error = np.nansum(
            bin_histogram / len(dataloader.dataset)
            * np.abs(bin_accuracies - bin_centres)
        )
        # quick error check
        assert np.allclose(
            accuracy.numpy(),
            np.nansum(bin_accuracies * bin_histogram)
            / len(dataloader.dataset)
        )
    return accuracy

@pytest.mark.parametrize(
    'dataset', ['cifar10-valid', 'cifar100', 'ImageNet16-120']
)
def test_evaluation_metrics(caplog, get_test_batch_size, dataset):
    """Compare computed evaluation metrics to NAS-bench."""
    caplog.set_level(logging.INFO)
    # Choose a random seed.
    seed = random.choice([777, 888])
    # Randomly select an architecture.
    index = random.randrange(0, 15625)
    # Load the architecture and its true metrics
    api = create(None, 'tss', fast_mode=True, verbose=False)
    hp = "200"
    api.reload(None, index)
    results = api.get_more_info(index, dataset, hp=hp, is_random=seed)
    # cost_info = api.get_cost_info(index, dataset)
    # (
    #     validation_accuracy, latency, time_cost, current_total_time_cost
    # ) = api.simulate_train_eval(index, dataset, hp=hp)
    config = api.get_net_config(index, dataset)
    network = get_cell_based_tiny_net(config)
    params = api.get_net_param(
        index, dataset, seed, hp=hp
    )
    network.load_state_dict(params)
    # Compute metrics
    dataset_name = (
        dataset if not '-valid' in dataset else dataset[:-6]
    )
    (
        train_dataloader, valid_dataloader, test_dataloader, class_num
    ) = get_data_splits(dataset_name, Path('../data'))
    network.train()
    train_accuracy = get_accuracy(train_dataloader, network)
    network.eval()
    valid_accuracy = get_accuracy(valid_dataloader, network) * 2
    test_accuracy = get_accuracy(test_dataloader, network)
    if dataset == 'cifar10-valid':
        train_accuracy = train_accuracy * 2 
    else:
        test_accuracy = test_accuracy * 2
    # Compare
    print(f'Train: {results["train-accuracy"]} == {train_accuracy.item()}')
    print(f'Valid: {results["valid-accuracy"]} == {valid_accuracy.item()}')
    print(f'Test: {results["test-accuracy"]} == {test_accuracy.item()}')
    assert results['train-accuracy'] == train_accuracy.item()
    assert results['valid-accuracy'] == valid_accuracy.item()
    assert results['test-accuracy'] == test_accuracy.item()
