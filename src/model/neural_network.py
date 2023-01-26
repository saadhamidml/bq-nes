"""Utilities for neural networks."""
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
import os.path
import logging
from networkx import Graph
import math
import numpy as np
import random
import torch
from torch.optim import SGD, swa_utils, lr_scheduler
from nats_bench.api_topology import NATStopology
from xautodl.models import get_cell_based_tiny_net


logger = logging.getLogger(__name__)


class AveragedModelWithVariance(swa_utils.AveragedModel):
    """Averaged Model that keeps track of parameter variances."""

    def __init__(self, model, device=None, avg_fn=None, std_fn=None):
        """Initialise.
        
        :param model: The neural network.
        :param device: The device on which to store the SWA model.
        :param avg_fn: Function for computing SWAG averages.
        :param std_fun: Function for computing SWAG standard deviations.
        """
        super().__init__(model, device=device, avg_fn=avg_fn)
        self.std_module = deepcopy(model)
        if device is not None:
            self.std_module = self.std_module.to(device)
        if std_fn is None:
            def std_fn(
                std_model_parameter,
                last_avg_model_parameter,
                cur_avg_model_parameter,
                model_parameter,
            ):
                return (
                    std_model_parameter
                    + (model_parameter - last_avg_model_parameter)
                    * (model_parameter - cur_avg_model_parameter)
                )
        self.std_fn = std_fn
        self.register_buffer(
            'log_model_evidence', torch.tensor(-np.inf, device=device)
        )
    
    def update_parameters(self, model):
        """Update the SWAG averages and standard deviations.
        
        :param model: The neural network.
        """
        for p_swa, p_std, p_model in zip(
            self.parameters(), self.std_module.parameters(), model.parameters()
        ):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
                p_std.detach()
                p_std.data = torch.zeros_like(p_model_)
            else:
                last_p_swa = deepcopy(p_swa)
                p_swa.detach().copy_(self.avg_fn(
                    p_swa.detach(), p_model_, self.n_averaged.to(device)
                ))
                p_std.detach().copy_(self.std_fn(
                    p_std.detach(),
                    last_p_swa.detach(),
                    p_swa.detach(),
                    p_model_
                ))
        self.n_averaged += 1


def train_network(
    network: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    seed: int,
    smoke_test: bool = False
) -> None:
    """Train a Neural Network using the NATS-Bench default
    hyperparameters for a new seed.
    
    :param network: The network.
    :param train_loader: The training data loader.
    :param seed: The new seed.
    :param smoke_test: Whether to run with fastest options (for
        debugging).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    network.cuda().train()
    num_epochs = 200
    optimiser = SGD(
        network.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        for train_inputs, train_targets in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(
                network(train_inputs.cuda())[1], train_targets.cuda()
            )
            loss.backward()
            optimiser.step()
        if smoke_test and epoch > 0:
            break
        scheduler.step()


def train_swa_gaussian(
    network: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    smoke_test: bool = False
) -> Tuple[torch.nn.Module, torch.Tensor]:
    """Use SWA-Gaussian to approximate the posterior over architecture
    weights.
    
    See https://arxiv.org/abs/1902.02476 for details.

    The approximate posterior is used to estimate the model evidence,
    but the final weights are set to the mean of the Gaussian.

    It is assumed that the initial weights have been pre-trained.

    :param network: The neural network. The weights are updated as a
        side-effect of this function.
    :param train_loader: DataLoader containing the training set.
    :param smoke_test: Whether to run with fastest options (for
        debugging).
    :return: The model evidence of the architecture.
    """
    network.cuda().train()
    # Optimiser consistent with pre-training. Note that learning rate
    # was Cosine annealed from 0.1 to 0.0 over 200 epochs.
    optimiser = SGD(
        network.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    swa_model = AveragedModelWithVariance(network, device=torch.device('cpu'))
    swa_scheduler = swa_utils.SWALR(
        optimiser,
        anneal_strategy='cos',  # To be consistent with pretraining.
        anneal_epochs=5,
        swa_lr=0.0  # To be consistent with pretraining.
    )
    num_epochs = 25  # So that we end up with 10 SWA samples
    loss_fn = torch.nn.CrossEntropyLoss()

    log_likelihoods = []
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        lls = []
        for train_inputs, train_targets in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(
                network(train_inputs.cuda())[1], train_targets.cuda()
            )
            loss.backward()
            lls.append(loss.detach().item())
            optimiser.step()
            if smoke_test:
                break
        if (epoch + 1) % 5 == 0:
            log_likelihoods.append(-np.sum(lls).item())
            swa_model.update_parameters(network)
        swa_scheduler.step()
    swa_utils.update_bn(train_loader, swa_model)
    swa_model.cuda()
    
    log_model_evidence = torch.tensor(0.).cuda()
    log_laplace_correction = torch.tensor(0.).cuda()
    with torch.no_grad():
        for train_inputs, train_targets in train_loader:
            log_model_evidence -= loss_fn(swa_model(
                train_inputs.cuda())[1], train_targets.cuda()
            )
            if smoke_test:
                break
        for p_std in swa_model.std_module.parameters():
            # Clamp min to ensure estimate is not reduced.
            log_laplace_correction += 0.5 * torch.log(
                (2 * math.pi * p_std.clamp_min(1 / (2 * math.pi))).pow(2)
            ).sum()
    if log_laplace_correction == 0:
        logger.warning('SWA-Gaussian variance underestimated.')
    log_model_evidence += log_laplace_correction
    # log_model_evidence_mc = (
    #     torch.logsumexp(torch.tensor(log_likelihoods), 0)
    #     - np.log(len(log_likelihoods))
    # )

    swa_model.log_model_evidence = log_model_evidence

    return swa_model


def query_swag_model_evidence(
    arch: Union[Graph, int],
    api: NATStopology,
    dataset: str,
    train_loader: torch.utils.data.DataLoader,
    hp: str = "200",
    seed: int = 777,
    data_dir: Path = Path('../data'),
    smoke_test: bool = False
) -> float:
    """Query model evidence using SWAG approximation. Log SWA weights to
    disk.
    
    :param arch: The architecture, either as a graph or as api index.
    :param api: The NATS-Bench topology API.
    :param dataset: The dataset.
    :param train_loader: DataLoader containing the training set.
    :param hp: The number of training epochs.
    :param seed: The NATS-Bench seed. If None then we do multi-SWAG.
    :param data_dir: Relative path to benchmark data.
    :param smoke_test: Run with fastest settings (for debugging).
    :return: (Approximate) log likelihood of the architecture.
    """
    if seed is None:
        raise NotImplementedError
    seed = [777, 888] if seed is None else seed

    # Load the trained model
    try:
        index = api.query_index_by_arch(arch.name)
    except AttributeError:
        index = arch
    config = api.get_net_config(index, dataset)
    network = get_cell_based_tiny_net(config)
    api.reload(None, index)
    params = api.get_net_param(
        index, dataset, seed=seed, hp=hp
    )
    if seed is None:
        params_iterable = iter(params.values())
        params = next(params_iterable)
    network.load_state_dict(params)

    # Check that model is not already saved.
    save_file = (
        data_dir / f'NATS-tss-v1_0-3ffb9-swa/{dataset}/{hp}/{seed}/{index}.pt'
    )

    if save_file.is_file():
        swa_model = AveragedModelWithVariance(network).cuda()
        swa_model.load_state_dict(torch.load(save_file))
    else:
        # Train to get SWA weight samples.
        swa_model = train_swa_gaussian(
            network, train_loader, smoke_test=smoke_test
        )
        torch.save(swa_model.state_dict(), save_file)

    return swa_model.log_model_evidence
