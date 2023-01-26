import logging
from pathlib import Path
from turtle import width
import numpy as np
import scipy.special
import torch
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append('../')

from bayes_quad.generate_test_graphs import random_sampling
from model.neural_network import query_swag_model_evidence
from utils.evaluation import get_data_splits
from utils.nas import get_architecture_log_likelihood, load_design_set
from utils.plotting import PLOT_PARAMS

from pp_utils import (
    sort_design_set_by_likelihood,
    rebuild_surrogate
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def plot_design_set(dataset, log_dir):
    """Plot the design set."""
    # Load ranked architectures.
    pickle = np.load(
        f'../../logs/sandbox/nas/{dataset}/ranked_architectures.npz'
    )
    ranking = pickle['ranking']
    valid_log_likelihoods = pickle['valid_log_likelihoods']

    # Load design set.
    log_dir_ = Path(f'../../logs/{log_dir}')
    with open(log_dir_ / 'archs.txt') as f:
        design_set = f.readlines()
    design_set = np.array(list(map(lambda x: int(x[:-1]), design_set)))
    
    # Encode design set as a mask of the ranked architectures.
    mask = ranking.reshape(1, -1) == design_set.reshape(-1, 1)
    design_set = np.sum(mask, axis=0)

    # Plot
    indices = np.arange(ranking.shape[0])
    likelihoods = np.exp(valid_log_likelihoods)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=indices,
            y=likelihoods * design_set,
            name='Design Set',
            width=25,
            # textposition='outside'
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=indices,
    #         y=likelihoods * (1 - design_set),
    #         name='Unsampled'
    #     ),
    # )
    fig.update_layout(
        uniformtext_minsize=8, uniformtext_mode='show', **PLOT_PARAMS
    )
    fig['layout']['xaxis']['title'] = 'Arch Ranking'
    fig['layout']['yaxis']['title'] = 'Likelihoods'
    fig.write_html(str(log_dir_ / f'design-set.html'))
    fig.write_image(str(log_dir_ / f'design-set.pdf'))


def plot_bq_components(dataset, log_dir):
    logger.info(f'Post Processing {log_dir}')
    api = create(None, 'tss', fast_mode=True, verbose=False)
    archs, log_likelihoods, ensemble_weights = load_design_set(
        api, dataset, Path(f'../../logs/{log_dir}')
    )
    archs, log_likelihoods, ensemble_weights = sort_design_set_by_likelihood(
        archs, log_likelihoods, ensemble_weights
    )
    integrand_model, _ = rebuild_surrogate(api, archs, log_likelihoods)

    K_i = integrand_model.surrogate.K_i.detach().numpy()
    fig = px.imshow(K_i)
    log_dir_ = Path(f'../../logs/{log_dir}')
    fig.write_html(str(log_dir_ / 'K-inverse-heatmap.html'))
    fig.write_image(str(log_dir_ / 'K-inverse-heatmap.pdf'))

    K_i_col_sums = K_i.sum(0)
    indices = np.arange(K_i_col_sums.shape[0])
    fig = go.Figure(data=go.Bar(x=indices, y=K_i_col_sums))
    fig.update_layout(**PLOT_PARAMS)
    fig.write_html(str(log_dir_ / 'K-inverse-column-sums.html'))
    fig.write_image(str(log_dir_ / 'K-inverse-column-sums.pdf'))

    samples, _, _ = random_sampling(
        pool_size=2048,
        benchmark='nasbench201'  # Same as NATS-Bench TSS
    )
    with torch.no_grad():
        int_kernel = (
            integrand_model.surrogate.cross_covariance(samples).sum(dim=1)
            / 2048
        ).numpy()
    fig = go.Figure(data=go.Bar(x=indices, y=int_kernel))
    fig.update_layout(**PLOT_PARAMS)
    fig.write_html(str(log_dir_ / 'kernel_integral.html'))
    fig.write_image(str(log_dir_ / 'kernel_integral.pdf'))


def plot_likelihood_surface(dataset):
    # Load ranked architectures.
    pickle = np.load(
        f'../../logs/sandbox/nas/{dataset}/ranked_architectures.npz'
    )
    design_set = pickle['ranking'].tolist()
    # log_likelihoods = pickle['valid_log_likelihoods']
    train_log_likelihoods = []
    approx_train_log_likelihoods = []
    valid_log_likelihoods = []
    test_accuracies = []
    for a in design_set:
        tlls = []
        vlls = []
        tas = []
        api = create(None, 'tss', fast_mode=True, verbose=False)
        results = api.get_more_info(a, dataset, hp='12', is_random=111)
        tll = -results['train-loss']
        if not np.isnan(tll):
            # tlls.append(tll)
            approx_train_log_likelihoods.append(tll)
        # vll = -results['valid-loss']
        # if not np.isnan(vll):
        #     vlls.append(vll)
        # ta = results['test-accuracy']
        # if not np.isnan(ta):
        #     tas.append(ta)
        seeds = [777, 888]
        for s in seeds:
            # Recreate api to save memory
            results = api.get_more_info(
                a, dataset, hp='200', is_random=s
            )
            tll = -results['train-loss']
            if not np.isnan(tll):
                tlls.append(tll)
            vll = -results['valid-loss']
            if not np.isnan(vll):
                vlls.append(vll)
            ta = results['test-accuracy']
            if not np.isnan(ta):
                tas.append(ta)
        train_log_likelihoods.append(
            scipy.special.logsumexp(tlls) - np.log(len(tlls))
        )
        valid_log_likelihoods.append(
            scipy.special.logsumexp(vlls) - np.log(len(vlls))
        )
        test_accuracies.append(np.mean(tas))
    train_log_likelihoods = np.array(train_log_likelihoods)
    approx_train_log_likelihoods = np.array(approx_train_log_likelihoods)
    valid_log_likelihoods = np.array(valid_log_likelihoods)
    test_accuracies = np.array(test_accuracies)
    sort_inds = np.argsort(test_accuracies)[::-1]
    fig = px.line(x=np.arange(train_log_likelihoods.shape[0]), y=np.exp(train_log_likelihoods[sort_inds]))
    fig.add_scatter(x=np.arange(valid_log_likelihoods.shape[0]), y=np.exp(valid_log_likelihoods[sort_inds]))
    fig.add_scatter(x=np.arange(approx_train_log_likelihoods.shape[0]), y=np.exp(approx_train_log_likelihoods[sort_inds]))
    fig.write_html('plot2_w_approx.html')


def plot_swag_likelihood(dataset: str):
    """Plot SWAG likelihood surface.
    
    :param dataset: The dataset.
    """
    # Load ranked archiectures, and select subset.
    pickle = np.load(
            f'../../logs/sandbox/nas/{dataset}/train_ranked_architectures.npz'
        )
    indices = 625 * np.arange(20)
    search_space = pickle['ranking'].tolist()
    design_set = pickle['ranking'][indices].tolist()
    
    data_dir = Path('../../data')
    seed = 777
    hp = '200'
    batch_size = None
    train_loader, *_ = get_data_splits(
        dataset, data_dir=data_dir, batch_size=batch_size
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    mle_lls = []
    swag_lls = []
    for api_index in design_set:
        api = create(None, 'tss', fast_mode=True, verbose=False)
        # config = api.get_net_config(api_index, dataset)
        # network = get_cell_based_tiny_net(config)
        # api.reload(None, api_index)
        # params = api.get_net_param(
        #     api_index, dataset, seed=seed, hp=hp
        # )
        # if seed is None:
        #     params_iterable = iter(params.values())
        #     params = next(params_iterable)
        # network.load_state_dict(params)
        # network.cuda()
        # # Query training loss.
        # train_log_likelihood = torch.tensor(0.).cuda()
        # with torch.no_grad():
        #     for train_inputs, train_targets in train_loader:
        #         train_log_likelihood -= loss_fn(
        #             network(train_inputs.cuda())[1], train_targets.cuda()
        #         )
        # mle_lls.append(train_log_likelihood.detach().item())
        mle_lls.append(
            get_architecture_log_likelihood(
                api_index, api, dataset, hp=hp, seed=seed
            )
            * len(train_loader.dataset) / train_loader.batch_size
        )
        swag_lls.append(query_swag_model_evidence(
            api_index, api, dataset, hp=hp, seed=seed, data_dir=data_dir
        ).detach().item())
    
    mle_lls = np.array(mle_lls)
    swag_lls = np.array(swag_lls)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=mle_lls, name='MLE'))
    fig.add_trace(go.Scatter(x=indices, y=swag_lls, name='SWAG'))
    log_dir = Path(f'../../logs/sandbox/likelihood_surface/{dataset}')
    log_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(log_dir / 'train_mle_swag.html'))


if __name__ == '__main__':
    # plot_likelihood_surface('cifar100')
    plot_swag_likelihood('ImageNet16-120')
