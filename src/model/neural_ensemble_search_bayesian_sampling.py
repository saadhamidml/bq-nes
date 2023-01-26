import logging
from typing import Mapping, Sequence, Tuple
from pathlib import Path
import functools
from networkx import Graph
import torch
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS
from nats_bench import NATStopology, create
from gpytorch.kernels import RBFKernel

from bayes_quad.generate_test_graphs import create_nasbench201_graph
from utils.evaluation import evaluate_ensemble
from utils.nas import arch_str_to_op_list, get_architecture_log_likelihood, get_data_splits, index_to_nx, rank_design_set, OPS


logger = logging.getLogger(__name__)


def neural_ensemble_search_bayesian_sampling(
    api: NATStopology,
    dataset: str,
    nas_options: Mapping,
    load_design_set: int,
    data_dir: Path = Path('../data'),
    ensemble_size: int = 3,
    iterations: int = 10,
    log_dir: Path = Path('./'),
    smoke_test: bool = False
) -> Tuple[Sequence[Graph], torch.Tensor]:
    """Neural Ensemble Search with Bayesian Sampling.
    
    :param ensemble_size: The ensemble size.
    :param iterations: The number of iterations.
    :param load_design_set: The Sacred Run ID from which to load the
        design set.
    :param log_dir: The loggin directory.
    :param smoke_test: Whether the run is a smoke test.
    :return: The ensemble architectures and their log likelihoods.
    """
    # Load the validation dataset.
    _, valid_loader, _, num_classes = get_data_splits(
        'cifar10' if dataset == 'cifar10-valid' else dataset,
        data_dir=data_dir, batch_size=nas_options.get('batch_size')
    )
    num_data = len(valid_loader.dataset) / 2  # Validation set is always half a DataLoader.
    with open(log_dir.parent / f'{load_design_set}/archs.txt') as f:
        archs = f.readlines()
        print(f)
    archs = list(map(lambda x: int(x[:-1]), archs))
    archs = [
        index_to_nx(api, a, hp=nas_options.get('train_epochs', "200"))
        for a in archs
    ]
    query_arch = functools.partial(
        get_architecture_log_likelihood, api=api, dataset=dataset, data_dir=data_dir
    )
    log_likelihoods = torch.tensor([query_arch(a) for a in archs])
    archs, log_likelihoods = rank_design_set(archs, log_likelihoods)
    alpha = optimise_variational_posterior(archs, log_likelihoods)
    best_set = svgd_rd(alpha, ensemble_size)
    best_ll, *_ = evaluate_ensemble(
        valid_loader,
        num_data,
        num_classes,
        best_set,
        api,
        dataset,
        split='valid',
        seed=nas_options.get('seed', 777),
        train_epochs=nas_options.get('train_epochs', "200"),
        weights=None,
        param_ensemble_size=1,
        param_weights='even',
        sum_space='probability',
        smoke_test=smoke_test
    )
    for i in range(iterations - 1):
        logger.info(f'Sample Batch {i + 2}/{iterations}')
        sample_set = svgd_rd(alpha, ensemble_size)
        sample_ll, *_ = evaluate_ensemble(
            valid_loader,
            num_data,
            num_classes,
            sample_set,
            api,
            dataset,
            split='valid',
            seed=nas_options.get('seed', 777),
            train_epochs=nas_options.get('train_epochs', "200"),
            weights=None,
            param_ensemble_size=1,
            param_weights='even',
            sum_space='probability',
            smoke_test=smoke_test
        )
        if sample_ll > best_ll:
            best_set = sample_set
            best_ll = sample_ll
    # Save the archs and subset.
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(str(log_dir / 'archs.txt'), 'a') as f:
        for a in archs:
            index = api.query_index_by_arch(a.name)
            f.write(f'{index}\n')
    with open(str(log_dir / 'archs_subset.txt'), 'a') as f:
        for a in best_set:
            index = api.query_index_by_arch(a.name)
            f.write(f'{index}\n')
    return best_set, torch.tensor([query_arch(a) for a in best_set])


def arch_str_to_node_op_matrix(
    arch_str: str
) -> torch.Tensor:
    """Convert arch_str to matrix of node by operation."""
    op_list = arch_str_to_op_list(arch_str)
    mat = torch.zeros((6, 5))
    for node in range(6):
        for col_idx, op in enumerate(OPS):
            if op == op_list[node]:
                mat[node, col_idx] = 1
    return mat


def sample_variational_posterior(
    alpha: torch.Tensor
) -> Graph:
    """Sample from the variational posterior.
    
    :param alpha: Variational parameters, shape [6, 5].
    :return: The graph representation of the architecture.
    """
    op_list = []
    for node in range(alpha.size(0)):
        samples = F.gumbel_softmax(logits=alpha[node], hard=True)
        op_list += samples
    op_list = [OPS[o] for o in op_list]
    return create_nasbench201_graph(op_list)


def variational_kl_div(
    alpha: torch.Tensor
) -> torch.Tensor:
    """KL divergence between variational posterior and the prior.
    
    This turns out just to be the sum of the entropies, since the prior
    is flat.
    
    :param alpha: The variational parameters.
    :return: The KL divergence.
    """
    kl = torch.tensor(0.)
    for row in alpha:
        kl += -Categorical(logits=row).entropy()
    return kl


def optimise_variational_posterior(
    archs: Sequence[Graph] = None,
    log_likelihoods: torch.Tensor = None
):
    """Optimise the variational posterior.
    
    :param archs: The trained architectures.
    :param log_likelihoods: The corresponding validation log
        likelihoods.
    """
    # Num incoming connections in a cell x Num operations.
    # Values are logits. Initialise so that high mass on good architectures.
    alpha = torch.zeros((6, 5))  # [C, O]
    model_evidence = log_likelihoods.logsumexp(0).exp()
    for a, ll in zip(archs, log_likelihoods):
        alpha += ll.exp() * arch_str_to_node_op_matrix(a.name) / model_evidence
    alpha -= alpha.mean(dim=1, keepdim=True)
    alpha.requires_grad_(True)
    archs_op_ind = torch.stack(
        [arch_str_to_node_op_matrix(a.name).argmax(1) for a in archs]
    )  # [N, C]
    # Define optimiser as in NES-BS paper.
    opt = Adam([alpha], lr=1e-2, betas=(0.9, 0.999), weight_decay=3e-4)
    # opt = LBFGS([alpha], max_iter=5000, line_search_fn='strong_wolfe')
    # Optimise for 20 epochs, as in NES-BS paper.
    def closure():
        opt.zero_grad()
        arch_probs = Categorical(logits=alpha).log_prob(archs_op_ind).sum(-1).exp()
        loss = -(arch_probs * log_likelihoods).sum() + variational_kl_div(alpha)
        loss.backward()
        return loss
    for i in range(20):
        opt.step(closure)
    alpha.requires_grad_(False)
    return alpha


def svgd_rd(
    alpha: torch.Tensor,
    ensemble_size: int = 3,
    diversity: float = -0.5,
    iterations: int = 1000,
):
    """SVGD with Regularised Diversity.
    
    :param alpha: The variational parameters.
    :param ensemble_size: The ensemble size.
    :param diversity: The diversity hyperparameter.
    :param iterations: The number of iterations.
    """
    # Same normals for each node. mix_components represents 6 components
    # of mixture of normals.
    num_connects = alpha.size(0)  # C
    num_ops = alpha.size(1)  # O
    one_hots = F.one_hot(torch.arange(num_ops)).to(torch.get_default_dtype())
    mix_components = Normal(one_hots, num_ops * torch.ones_like(one_hots))
    variational = Categorical(logits=alpha)
    # Sample initial particles
    particles = []
    for i in range(ensemble_size):
        particle_ops = variational.sample()  # [C]
        particle = []
        for o in particle_ops:
            particle.append(mix_components.sample()[o])
        particle = torch.cat(particle, 0)
        particles.append(particle)
    particles = torch.stack(particles, 0)  # [N, C * O]
    particles.requires_grad_(True)
    # Optimise
    opt = SGD([particles], lr=0.1, momentum=0.9)
    kernel = RBFKernel()
    kernel.requires_grad_(False)
    def particle_log_probs(particles):
        log_probs = []
        for p in particles:
            p_ = p.view(num_connects, 1, num_ops)  # [C, 1, O]
            mix_log_probs =  mix_components.log_prob(p_).sum(-1)  # [C, O]
            log_prob = (variational.probs.log() + mix_log_probs).logsumexp(1).sum(0)
            log_probs.append(log_prob)
        return torch.stack(log_probs, 0)
    for i in range(iterations):
        opt.zero_grad()
        kl_term = num_connects * diversity * kernel(particles).evaluate().mean() 
        ker_term = -particle_log_probs(particles).mean()
        loss = kl_term + ker_term
        loss.backward()
        opt.step()
    # Turn particles to ensembles.
    ensemble = []
    for p in particles:
        p_ = p.view(num_connects, 1, num_ops)  # [C, 1, O]
        dists = (p_ - one_hots).pow(2).sum(-1)  # [C, O]
        ops = dists.argmin(-1)  # [C]
        arch = create_nasbench201_graph([OPS[o] for o in ops])
        ensemble.append(arch)
    return ensemble
