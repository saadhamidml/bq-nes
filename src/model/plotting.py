from pathlib import Path
import numpy as np
from plotly.subplots import make_subplots
from plotly import graph_objects as go

from utils.plotting import PLOT_PARAMS


def plot_ensemble_weights(
        log_likelihoods: np.ndarray,
        ensemble_weights:np.ndarray,
        log_dir: Path = Path('./')
):
    """Sort by log likelihood and plot likelihoods and ensemble weights.
    
    :param log_likelihoods: The log_likelihoods, shape [N].
    :param ensemble_weights: The ensemble weights, shape [N].
    :param log_dir: The directory to save the plots to.
    """
    sort_inds = np.argsort(log_likelihoods)[::-1]
    likelihoods = np.exp(log_likelihoods[sort_inds])
    ensemble_weights = ensemble_weights[sort_inds]
    indices = np.arange(ensemble_weights.shape[0])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Bar(x=indices, y=likelihoods, name='Likelihoods'),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=indices, y=ensemble_weights, name='Ensemble Weights'
        ),
        row=2,
        col=1
    )
    fig.update_layout(**PLOT_PARAMS)
    fig['layout']['xaxis2']['title'] = 'Arch Index'
    fig['layout']['yaxis']['title'] = 'Likelihoods'
    fig['layout']['yaxis2']['title'] = 'Ensemble Weights'
    fig.write_html(str(log_dir / f'ensemble-weights.html'))
    fig.write_image(str(log_dir / f'ensemble-weights.pdf'))
