"""Utils for exploring the NAS search space."""
from pathlib import Path
from turtle import width
from typing import Mapping, Sequence, Tuple, List, Union
import logging
from copy import deepcopy
import itertools
import functools
import random
import math
import numpy as np
import scipy.special
import torch
from nats_bench import create
from nats_bench.api_topology import NATStopology
from networkx import Graph
from PIL import Image
from xautodl.config_utils import load_config
from xautodl.datasets import get_datasets
from xautodl.models import get_cell_based_tiny_net
import plotly.express as px

import sys; sys.path.append('../')

from utils.plotting import PLOT_PARAMS

from kernels.weisfilerlehman import WeisfilerLehman
from bayes_quad.gp import GraphGP
from bayes_quad.generate_test_graphs import create_nasbench201_graph
from model.neural_network import query_swag_model_evidence


logger = logging.getLogger(__name__)


OPS = [
    'none',
    'skip_connect',
    'avg_pool_3x3',
    'nor_conv_1x1',
    'nor_conv_3x3',
]
BATCHING_FACTORS = {
    'cifar10-valid': {
        'train': 25000 / 256,
        'valid': 25000 / 256,
        'test': 10000 / 256
    },
    'cifar100': {
        'train': 50000 / 256,
        'valid': 5000 / 256,
        'test': 5000 / 256
    },
    'ImageNet16-120': {
        'train': 151700 / 256,
        'valid': 3000 / 256,
        'test': 3000 / 256
    },
}
VALID_BATCH_SIZES = {
    'cifar10-valid': 10000,
    'cifar100': 5000,
    'ImageNet16-120': 3000
}


def get_data_splits(
        dataset: str, data_dir: Path = Path('../data'), batch_size: int = None
) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        int
]:
    """Get the train/valid/test data splits, and the number of classes.
    
    :param dataset: The dataset.
    :param data_dir: The relative path to the benchmark data.
    :param batch_size: The batch size for the dataloaders.
    :return: Train dataloader, validation dataloader, test dataloader,
        number of classes.
    """
    # look all the datasets
    # load the configuration
    if dataset == "cifar10" or dataset == "cifar100":
        dataset_directory = 'cifar.python'
        split_info = load_config(
            f"{data_dir}/configs/cifar-split.txt", None, None
        )
    elif dataset.startswith("ImageNet16"):
        dataset_directory = 'cifar.python/ImageNet16'
        split_info = load_config(
            f"{data_dir}/configs/{dataset}-split.txt", None, None
        )
    else:
        raise ValueError("invalid dataset : {:}".format(dataset))
    # train valid data
    train_data, valid_data, xshape, class_num = get_datasets(
        dataset, str((data_dir / dataset_directory).resolve()), -1
    )
    if batch_size is None:
        config = load_config(
            f'{data_dir}/configs/200E.config',
            dict(class_num=class_num, xshape=xshape),
            None
        )
        batch_size = config.batch_size
    if dataset == "cifar10":
        ValLoaders = {
            "ori-test": torch.utils.data.DataLoader(
                valid_data,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
        }
        assert len(train_data) == len(split_info.train) + len(
            split_info.valid
        ), "invalid length : {:} vs {:} + {:}".format(
            len(train_data), len(split_info.train), len(split_info.valid)
        )
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                split_info.train
            ),
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                split_info.valid
            ),
            pin_memory=True,
        )
        ValLoaders["x-valid"] = valid_loader
        return (
            train_loader,
            ValLoaders['x-valid'],
            ValLoaders['ori-test'],
            class_num
        )
    else:
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        if dataset == "cifar100":
            cifar100_splits = load_config(
                f"{data_dir}/configs/cifar100-test-split.txt", None, None
            )
            ValLoaders = {
                "ori-test": valid_loader,
                "x-valid": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xvalid
                    ),
                    pin_memory=True,
                ),
                "x-test": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xtest
                    ),
                    pin_memory=True,
                ),
            }
        elif dataset == "ImageNet16-120":
            imagenet16_splits = load_config(
                f"{data_dir}/configs/imagenet-16-120-test-split.txt",
                None,
                None
            )
            ValLoaders = {
                "ori-test": valid_loader,
                "x-valid": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xvalid
                    ),
                    pin_memory=True,
                ),
                "x-test": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xtest
                    ),
                    pin_memory=True,
                ),
            }
        else:
            raise ValueError("invalid dataset : {:}".format(dataset))
        return (
            train_loader,
            ValLoaders['x-valid'],
            ValLoaders['x-test'],
            class_num
        )

def arch_str_to_op_list(arch_str: str) -> List[str]:
    """Architecture as a string to a list of operations.
    
    :param arch_str: The architecture as a string.
    :return: The architecture as a list of operations.
    """
    op_list = list(filter(lambda i: (i != '' and i != '+'), arch_str.split('|')))
    return list(map(lambda x: x[:-2], op_list))


def prune_arches(archs: Sequence[Graph]) -> Sequence[Graph]:
    """Remove invalid architectures.
    
    :param archs: The list of architectures as graphs.
    :return: The pruned list.
    """
    return [a for a in archs if len(a) != 0 and a.number_of_edges() != 0]


def index_to_nx(
    api: NATStopology, index: int, hp: str = "200"
) -> Union[Graph, None]:
    """NATS-Bench index to graph representation.
    
    Returns None if the architecture is invalid.

    :param api: NATS-Bench API.
    :param index: Architecture API index.
    :param hp: Number of training epochs.
    :return: Graph representation of architecture, or None if
        architecture is invalid.
    """
    op_labelling = arch_str_to_op_list(
        api.query_meta_info_by_index(index, hp=hp).arch_str
    )
    graph = create_nasbench201_graph(op_labelling)
    if len(graph) == 0 or graph.number_of_edges() == 0:
        # IN Nasbench201, it is possible that invalid graphs consisting
        # entirely from None and skip-line are generated; remove these invalid
        # architectures.

        # Also remove if the number of edges is zero. This is is possible, one
        # example in NAS-Bench-201:
        # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|none~2|'
        return None
    else:
        return graph


def random_sampling(
    n_samples: int,
    api,
    dataset,
    hp: str = "200",
    restrict_to_best: int = None,
    log_dir: Path = None
) -> List[Graph]:
    """Sample architectures.
    
    :param n_samples: Number of samples. Note that invalid architectures
        are dropped if they're sampled.
    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    :param hp: The number of training epochs.
    :param restrict_to_best: Whether to restrict the search space to the
        best m architectures (as measured by valid-loss with seed 777
        and 200 training epochs).
    :param log_dir: The logs directory.
    :return: List of sampled architectures.
    """
    if log_dir is not None:
        pickle = np.load(
            f'{log_dir}/sandbox/nas/{dataset}/ranked_architectures.npz'
        )
        architecture_space = pickle['ranking'].tolist()
    else:
        architecture_space = list(range(15_625))
    if restrict_to_best:
        architecture_space = architecture_space[:restrict_to_best]
    if n_samples == len(architecture_space):
        # Just return the whole space.
        # This is for acquisition function optimisation.
        sample_indices = architecture_space
    else:
        sample_indices = random.choices(architecture_space, k=n_samples)
    samples = [index_to_nx(api, i, hp=hp) for i in sample_indices]
    return [g for g in samples if g is not None]


def unique_random_samples(
    n_samples: int,
    api: NATStopology,
    dataset: str,
    hp: str = "200",
    **kwargs
) -> List[Graph]:
    """Sample architectures without replacement. If an invalid
    architecture is sampled then we drop it and keep sampling until we
    have n_samples valid architectures.

    :param n_samples: The number of samples.
    :param api: The api.
    :param dataset: The dataset.
    :param hp: The number of training epochs.
    :return: The sampled architectures.
    """
    samples = []
    while len(samples) < n_samples:
        sample = random_sampling(1, api, dataset, hp=hp)
        try:
            if all(sample[0].name != l.name for l in samples):
                samples += sample
        except (AttributeError, IndexError) as e:
            pass
    return samples


def get_architecture_log_likelihood(
    arch: Union[Graph, int],
    api: NATStopology,
    dataset: str,
    dataset_split: str = 'valid',
    hp: str = "200",
    seed: int = 777,
    data_dir: Path = Path('../data'),
    batching_correction: float = None,
    swag: bool = False,
) -> float:
    """Get the log likelihood of the architecture.
    
    :param arch: The architecture (graph representation or API index).
    :param api: The NATS-Bench API.
    :param dataset: The dataset.
    :param dataset_split: Whether to use the "train" set or "valid" set.
    :param hp: The number of training epochs.
    :param seed: The NAS seed. If None, then we average over the two
        available seeds for 200 epochs (777 and 888).
    :param data_dir: Relative path to benchmark data.
    :param batching_correction: If precomputed, provide the
        batching_correction. (The cost of loading the dataloaders adds
        up if getting the likelihoods for lots of architectures.)
    :param swag: Whether to use SWAG to approximate the log likelihood.
    :return: (Approximate) log likelihood of the architecture.
    """
    if batching_correction is None:
        batching_correction = BATCHING_FACTORS[dataset][dataset_split]
        if batching_correction is None:
            train_loader, valid_loader, *_ = get_data_splits(
                dataset, data_dir=data_dir
            )
            split_loader = (
                valid_loader if dataset_split == 'valid' else train_loader
            )
            batching_correction = (
                len(split_loader.dataset) / split_loader.batch_size
            )
    try:
        index = api.query_index_by_arch(arch.name)
    except AttributeError:
        index = arch
    seed = [777, 888] if seed is None else seed
    if not swag:
        if seed is None:
            lls = []
            for s in seed:
                ll = -api.get_more_info(index, dataset, hp=hp, is_random=s)[
                    f'{dataset_split}-loss'
                ] * batching_correction
                if not np.isnan(ll):
                    lls.append(ll)
            log_likelihood = (
                scipy.special.logsumexp(lls) - np.log(len(lls))
            ).item()
        else:
            log_likelihood = -api.get_more_info(
                index, dataset, hp=hp, is_random=seed
            )[f'{dataset_split}-loss'] * batching_correction
    else:
        log_likelihood = query_swag_model_evidence(
            arch,
            api,
            dataset,
            train_loader,
            hp=hp,
            seed=seed,
            data_dir=data_dir
        )
    return log_likelihood


def load_architecture(
    arch: Union[Graph, int],
    api: NATStopology,
    dataset: str,
    seed: int = None,
    hp: str = "200"
) -> Tuple[torch.nn.Module, Union[Mapping, Mapping]]:
    """Load an architecture.
    
    :param arch: The architecture graph or api index.
    :param api: The api.
    :param dataset: The dataset.
    :param seed: The seed.
    :param hp: The number of training epochs.
    :return: The network, along with a dictionary of parameter settings
        if seed is None, else None.
    """
    try:
        api_index = api.query_index_by_arch(arch.name)
    except AttributeError:
        api_index = arch
    config = api.get_net_config(api_index, dataset)
    network = get_cell_based_tiny_net(config)
    api.reload(None, api_index)
    params_dict = api.get_net_param(api_index, dataset, seed=seed, hp=hp)
    api.clear_params(api_index)
    if seed is None:
        params = params_dict[777]
    else:
        params = params_dict
        params_dict = {seed: params_dict}
    network.load_state_dict(params)
    return network, params_dict


def load_design_set(
    api: NATStopology,
    dataset: str,
    log_dir: Path,
    hp: str = "200",
    seed: int = 777
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Load design set (saved as list of API indices) from file.
    
    :param api: NATS-Bench API.
    :param dataset: The dataset.
    :param log_dir: The logging directory containing the saved info.
    :param hp: The number of training epochs.
    :param seed: The NATS-Bench seed.
    :return: The design set (as list of API indices), the corresponding
        log likelihoods, and previously calculated ensemble weights.
    """
    with open(log_dir / 'archs.txt') as f:
        archs = f.readlines()
    archs = list(map(lambda x: int(x[:-1]), archs))
    try:
        ensemble_weights = np.load(log_dir / 'ensemble-weights.npy')
    except FileNotFoundError as e:
        ensemble_weights = np.ones(len(archs)) / len(archs)

    log_likelihoods = np.array([
        -api.get_more_info(a, dataset, hp=hp, is_random=seed)['valid-loss']
        for a in archs
    ])

    sort_inds = np.argsort(log_likelihoods)
    archs = [archs[i] for i in sort_inds]
    log_likelihoods = log_likelihoods[sort_inds]
    ensemble_weights = ensemble_weights[sort_inds]

    return archs, log_likelihoods, ensemble_weights


def rank_design_set(
    archs: Sequence[Graph], likelihoods: torch.Tensor
) -> Tuple[Sequence[Graph], torch.Tensor]:
    """Rank the design set by likelihood (ascending).
    
    :param archs: The architectures.
    :param likelihoods: The corresponding likelihoods, shape [N].
    :return: The archs ranked (ascending) by likelihood, and sorted
        likelihoods.
    """
    likelihoods, sort_inds = torch.sort(likelihoods)
    archs = [archs[i] for i in sort_inds]
    return archs, likelihoods


def rank_architectures(
    dataset: str, log_dir: Path, criterion: str = 'valid'
) -> List[int]:
    """Rank architectures by loss. (Best first.)
    
    :param dataset: The dataset.
    :param log_dir: Directory to save results to.
    :param criterion: The criterion to use for ranking -- train, valid,
        or test.
    """
    lls = []
    num_archs = 15625
    for i in range(num_archs):
        # Reload api every time for memory reasons.
        api = create(None, 'tss', fast_mode=True, verbose=True)
        results = api.get_more_info(
            i, dataset, hp=200, is_random=777
            )
        lls.append(-results[f'{criterion}-loss'])
        api.clear_params(i)
    lls = np.array(lls)
    sort_inds = np.argsort(lls)[::-1]
    log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        log_dir / f'{criterion}_ranked_architectures.npz',
        ranking=sort_inds,
        log_likelihoods=lls[sort_inds]
    )


def check_local_minima(
    api: NATStopology,
    dataset: str,
    api_index: int,
    surrogate: GraphGP = None
):
    """Check if the architecture is a local minima of the validation
    loss.
    
    :param api: The NATS-Bench API.
    :param api_index: The API index.
    """
    # Load in the architecture as op list.
    arch = arch_str_to_op_list(
        api.query_meta_info_by_index(api_index, hp="200").arch_str
    )
    if surrogate is None:
        valid_loss = api.get_more_info(
            api_index, dataset, hp="200", is_random=777
        )['valid-loss']
    else:
        arch_graph = index_to_nx(api, api_index, hp="200")
        valid_lik, _ = surrogate.predict(arch_graph)
        valid_loss = -valid_lik
    # Enumerate over neighbours
    neighbour = deepcopy(arch)
    local_minima = True
    for edge, op in itertools.product(range(len(arch)), OPS):
        if arch[edge] != op:
            neighbour[edge] = op
            index = api.query_index_by_arch(
                create_nasbench201_graph(neighbour).name
            )
            if surrogate is None:
                neighbour_valid_loss = api.get_more_info(
                    index, dataset, hp="200", is_random=777
                )['valid-loss']
            else:
                neighbour_graph = index_to_nx(api, index, hp="200")
                neighbour_lik, _ = surrogate.predict(neighbour_graph)
                neighbour_valid_loss = -neighbour_lik
            if neighbour_valid_loss < valid_loss:
                local_minima = False
                break
        neighbour = deepcopy(arch)
    return local_minima


def count_local_minima(
    api: NATStopology,
    dataset: str,
    data_dir: Path = Path('../../data'),
    check_top_n: int = 100
):
    """Count the number of architectures that are local minima of the
    validation loss in the top n.
    
    :param api: The NATS Bench API.
    :param dataset: The dataset.
    :param check_top_n: The number of architectures to check.
    """
    # Load the ranking.
    pickle = np.load(
        f'{data_dir}/rankings/{dataset}/valid_ranked_architectures.npz'
    )
    ranking = pickle['ranking'].tolist()[:500]
    # Prepare surrogate
    locations = [index_to_nx(api, a, hp="200") for a in ranking]
    query_arch = functools.partial(
        get_architecture_log_likelihood,
        api=api,
        dataset=dataset,
        hp="200",
        seed=777,
        data_dir=data_dir
    )
    log_likelihoods = torch.tensor([query_arch(l) for l in locations])
    log_offset = log_likelihoods.max()
    shifted_log_likelihoods = log_likelihoods - log_offset
    scaled_likelihoods = shifted_log_likelihoods.exp()
    kernel = [WeisfilerLehman()]
    surrogate_model = GraphGP(
        locations,
        scaled_likelihoods,
        kernel,
        perform_y_normalization=True,
        log_offset=log_offset
    )
    # surrogate_model.fit(wl_subtree_candidates=(3,), max_lik=1e-2)
    surrogate_model.fit(max_lik=1e-1)
    # Iterate over the top n architectures.
    local_minima = []
    for i, arch in enumerate(ranking[:check_top_n]):
        logger.info(f'Checking Arch {i}')
        local_minima.append(check_local_minima(api, dataset, arch, surrogate_model))
    local_minima = np.array(local_minima)
    np.save(f'local_minima_{dataset}.npy', local_minima)
    print(f'{local_minima.sum()} Local Minima in Top {check_top_n} Archs')


def visualise_top_archs(
    api: NATStopology,
    dataset: str,
    data_dir: Path = Path('../../data'),
    check_top_n: int = 500
):
    pickle = np.load(
        f'{data_dir}/rankings/{dataset}/valid_ranked_architectures.npz'
    )
    ranking = pickle['ranking'].tolist()[:check_top_n *2]
    locations = [index_to_nx(api, a, hp="200") for a in ranking]
    locations = [l for l in locations if l is not None]
    query_arch = functools.partial(
        get_architecture_log_likelihood,
        api=api,
        dataset=dataset,
        hp="200",
        seed=777,
        data_dir=data_dir
    )
    log_likelihoods = torch.tensor([query_arch(l) for l in locations])
    log_offset = log_likelihoods.max()
    shifted_log_likelihoods = log_likelihoods - log_offset
    scaled_likelihoods = shifted_log_likelihoods.exp()
    kernel = [WeisfilerLehman()]
    surrogate_model = GraphGP(
        locations,
        scaled_likelihoods,
        kernel,
        perform_y_normalization=True,
        log_offset=log_offset
    )
    surrogate_model.fit(max_lik=1e-1)
    means, _ = surrogate_model.predict(locations)
    sort_inds = torch.argsort(means, descending=True)
    K = surrogate_model.K[sort_inds][:, sort_inds].detach().numpy()
    indices = []
    for i in range(len(K) - 1):
        if K[i, i + 1] < 1.0:
            indices.append(i)
    K = K[indices][:, indices]
    K = K[:check_top_n][:, :check_top_n]
    fig = px.imshow(K, zmin=0., zmax=1.)
    fig.update_layout(
        height=PLOT_PARAMS['width'] / 2,
        width=PLOT_PARAMS['width'] / 2,
        margin=dict(l=5,r=5,b=5,t=5),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.write_html(f'K_mat_{dataset}_.html')
    fig.write_image(f'K_mat_{dataset}_.pdf')
    fig.write_image(f'K_mat_{dataset}_.pdf')
    print(np.linalg.det(K))
    print(K.sum())


if __name__ == '__main__':
    # DATASET = 'cifar10-valid'
    # LOG_DIR = Path('../logs/sandbox/nas')
    # LOG_DIR.mkdir(parents=True, exist_ok=True)
    # CRITERION = 'valid'

    # rank_architectures(DATASET, LOG_DIR / DATASET, criterion=CRITERION)

    api = create(None, 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar100'
    # count_local_minima(api, dataset)
    visualise_top_archs(api, dataset)
    dataset = 'ImageNet16-120'
    # count_local_minima(api, dataset)
    visualise_top_archs(api, dataset)