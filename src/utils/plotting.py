"""Utilities for plotting"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Provide params for plotly.
venue = 'aistats'
# width in px
width = {'neurips': 530, 'aistats': 235}
font_family = {'neurips': 'Times New Roman', 'aistats': 'Times New Roman'}
# 1 smaller than text font size.
font_size = {'neurips': 9, 'aistats': 9}
aspect_ratio = 4/3
colours = ['#d66b5e', '#ff7f00', '#ab678a', '#7b6ea7', '#377eb8']
PLOT_PARAMS = {
    'width': width[venue],
    'height': int(width[venue] / aspect_ratio),
    'margin': {'l': 0, 'r':0, 't':0, 'b':0},
    'font': {'family': font_family[venue], 'size': font_size[venue]},
    'colorway': colours
}

def plot_surrogate_metrics(
    error: np.ndarray,
    stddev: np.ndarray,
    log_dir: Path = Path('./'),
    split: str = 'test'
):
    """Plot surrogate error and posterior standard deviation.
    
    :param error: The error (target - posterior mean).
    :param stddev: The posterior standard deviation.
    :param log_dir: The logging directory.
    :param split: Whether the train metrics or test metrics were input.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=np.zeros_like(stddev),
        error_y=dict(type='data', array=stddev, visible=True),
        name='Posterior StdDev'
    ))
    fig.add_trace(go.Scatter(y=error, name='Error'))
    fig.update_layout(
        xaxis_title='Test Set Index',
        yaxis_title='Likelihood Error',
        **PLOT_PARAMS
    )
    fig.write_html(str(log_dir / f'surrogate-{split}-performance.html'))
    fig.write_image(str(log_dir / f'surrogate-{split}-performance.pdf'))


def plot_calibration(
        bin_centres: np.ndarray,
        bin_accuracies: np.ndarray,
        bin_histogram: np.ndarray,
        log_dir: Path = Path('./'),
        prefix: str = '',
):
    # Calibration
    fig = go.Figure(
        data=go.Scatter(
            x=[0, 1], y=[0, 1], name='Ideal', line={'dash': 'dot'}
        )
    )
    fig.add_trace(
        go.Scatter(x=bin_centres, y=bin_accuracies, name='Model')
    )
    fig.update_layout(
        xaxis_title='Model Confidence',
        yaxis_title='Accuracy',
        **PLOT_PARAMS
    )
    fig.update_xaxes(automargin=True, range=[0, 1])
    fig.update_yaxes(automargin=True, range=[0, 1])
    fig.write_html(str(log_dir / f'{prefix}-calibration.html'))
    fig.write_image(str(log_dir / f'{prefix}-calibration.pdf'))
    # Histogram
    fig = go.Figure(
        data=go.Bar(x=bin_centres, y=bin_histogram)
    )
    fig.update_layout(
        xaxis_title='Model Confidence',
        yaxis_title='Frequency',
        **PLOT_PARAMS
    )
    fig.write_html(str(log_dir / f'{prefix}-histogram.html'))
    fig.write_image(str(log_dir / f'{prefix}-histogram.pdf'))


def plot_calibration_comparison(
        bin_centres,
        ensemble_acc,
        even_acc,
        best_acc,
        ensemble_hist,
        even_hist,
        best_hist,
        log_dir: Path = Path('./')
):
    # Calibration
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=bin_centres, y=ensemble_acc, name='BQ-Ensemble')
    )
    fig.add_trace(
        go.Scatter(x=bin_centres, y=even_acc, name='Even-Ensemble')
    )
    fig.add_trace(
        go.Scatter(x=bin_centres, y=best_acc, name='Best-Valid')
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Ideal',
            line={'dash': 'dot'}
        )
    )
    fig.update_layout(
        xaxis_title='Model Confidence',
        yaxis_title='Accuracy',
        **PLOT_PARAMS
    )
    fig.update_xaxes(automargin=True, range=[0, 1])
    fig.update_yaxes(automargin=True, range=[0, 1])
    fig.write_html(str(log_dir / f'calibration-comparison.html'))
    fig.write_image(str(log_dir / f'calibration-comparison.pdf'))
    # Histogram
    fig = go.Figure(
        data=[
            go.Bar(x=bin_centres, y=ensemble_hist, name='BQ-Ensemble'),
            go.Bar(x=bin_centres, y=even_hist, name='Even-Ensemble'),
            go.Bar(x=bin_centres, y=best_hist, name='Best-Valid'),
        ]
    )
    fig.update_layout(
        barmode='group',
        xaxis_title='Model Confidence',
        yaxis_title='Frequency',
        **PLOT_PARAMS
    )
    fig.write_html(str(log_dir / f'histogram-comparison.html'))
    fig.write_image(str(log_dir / f'histogram-comparison.pdf'))
