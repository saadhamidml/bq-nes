import logging
from pathlib import Path
import math
import numpy as np
from nats_bench import create
from nats_bench.api_topology import NATStopology

import sys; sys.path.append('../src')
from utils.nas import get_data_splits, get_architecture_log_likelihood


logger = logging.getLogger(__name__)


def compute_true_log_model_evidence(
    api: NATStopology,
    dataset: str,
    hp: str = "200",
    seed: int = 777,
    data_dir: Path = Path('../data')
) -> float:
    """Compute the true log model evidence.
    
    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    :param hp: The number of training epochs.
    :param seed: The NAS seed. If None, then we average over the two
        available seeds for 200 epochs (777 and 888).
    :param data_dir: Relative path to benchmark data.
    :return: The log model evidence.
    """
    _, valid_loader, *_ = get_data_splits(
        dataset if dataset != 'cifar10-valid' else 'cifar10',
        data_dir=data_dir
    )
    batching_correction = (
        len(valid_loader.dataset) / valid_loader.batch_size
    )
    num_archs = 15625
    log_model_evidence = -np.inf
    for i in range(num_archs):
        log_model_evidence = np.logaddexp(
            log_model_evidence, get_architecture_log_likelihood(
                i,
                api,
                dataset,
                hp=hp,
                seed=seed,
                data_dir=data_dir,
                batching_correction=batching_correction
            )
        )
        api.clear_params(i)
    log_model_evidence -= math.log(num_archs)
    log_model_evidence = log_model_evidence.item()
    logger.info(f'True Log Model Evidence: {log_model_evidence}')
    return log_model_evidence


if __name__ == '__main__':
    api = create(None, 'tss', fast_mode=True, verbose=True)
    dataset = 'cifar10-valid'
    hp = "200"
    seed = 777
    data_dir = Path('../data')

    log_model_evidence = compute_true_log_model_evidence(
        api, dataset, hp=hp, seed=seed, data_dir=data_dir
    )
    print(f'True Log Model Evidence for {dataset}: {log_model_evidence}')
