import itertools
import logging
from pathlib import Path
from typing import Dict
import pytest
from nats_bench import create
import torch

import sys
sys.path.append('../src')

from model.arch_ensemble import ArchEnsemble
from utils.evaluation import TEST_BATCH_SIZE
from utils.nas import get_architecture_log_likelihood

logger = logging.getLogger(__name__)

@pytest.fixture
def get_model_evidence() -> Dict[str, float]:
    """Get true model evidence.
    
    Use utils.nas.compute_model_evidence to verify numbers.
    
    :return: Dictionary of log model evidences, indexed by dataset.
    """
    log_model_evidence = {
        'cifar10-valid': -83.65742457546568,
        'cifar100': -50.04335642747517,
        'ImageNet16-120': -53.36738665227185
    }
    return log_model_evidence


@pytest.mark.parametrize(
    # 'dataset', ['cifar10-valid', 'cifar100', 'ImageNet16-120']
    'dataset', ['cifar100', 'ImageNet16-120']
)
@pytest.mark.parametrize(
    'warping', [None, 'linearised-sqrt', 'moment-matched-log']
)
# @pytest.mark.parametrize('warping', [None])
def test_bayesian_quadrature(caplog, get_model_evidence, dataset, warping):
    """Test Bayesian Quadrature for Architecture Model Evidence.
    
    :param caplog: PyTest logging fixture, or None if not using PyTest.
    :param get_model_evidence: PyTest fixture containing model
        evidences.
    :param dataset: The dataset.
    :param warping: The warping.
    """
    try:
        caplog.set_level(logging.INFO)
    except Exception as e:
        pass
    true_log_model_evidence = get_model_evidence[dataset]
    api = create(None, 'tss', fast_mode=True, verbose=False)
    ensemble = ArchEnsemble(
        dataset=dataset,
        api=api,
        data_dir=Path('../data')
    )
    acquisition_strategy = (
        'random' if warping is None else 'uncertainty_sampling'
    )
    # if dataset == 'cifar100':
    #     load_id = 1002
    # elif dataset == 'ImageNet16-120':
    #     load_id = 1273
    numerics = {
        'method': 'bayesian_quadrature',
        'initialisation': {'num_samples': 10},
        'evaluation_budget': 150,
        'surrogate': {'warping': warping},
        'acquisition': {'strategy': acquisition_strategy, 'num_initial': 1024},
        # 'acquisition': {'strategy': 'load', 'load_id': load_id},
        'kernel_integration': {'num_samples': 2048, 'batch_size': None},
        'sum_space': 'logit',
        'nas_options': {'seed': 777, 'train_epochs': 200},
        'test_batch_size': TEST_BATCH_SIZE[dataset],
    }
    log_dir = Path('../logs/tests')
    ensemble.learn(
        numerics=numerics,
        log_dir=log_dir,
        sacred_run=None
    )
    bq_log_model_evidence = ensemble._log_model_evidence.item()
    print(
        f'Even Log Model Evidence: {torch.logsumexp(ensemble._log_likelihoods, 0).item()}'
    )
    print(f'BQ Log Model Evidence: {bq_log_model_evidence}')
    error = abs(true_log_model_evidence - bq_log_model_evidence)
    print(f'Log Model Evidence Error: {error}')
    assert error < 1e-1
