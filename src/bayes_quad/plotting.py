from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.plotting import PLOT_PARAMS

    
def plot_kernel_inverse(integrand_model, log_dir: Path = Path('../logs')):
    """Plot surrogate's kernel's inverse, and its column sums.
    
    :param integrand_model: The integrand model.
    :param log_dir: The logging directory.
    """
    K_i = integrand_model.surrogate.K_i.detach().numpy()
    fig = px.imshow(K_i)
    fig.write_html(str(log_dir / 'K-inverse-heatmap.html'))
    fig.write_image(str(log_dir / 'K-inverse-heatmap.pdf'))

    K_i_col_sums = K_i.sum(0)
    indices = np.arange(K_i_col_sums.shape[0])
    fig = go.Figure(data=go.Bar(x=indices, y=K_i_col_sums))
    fig.update_layout(**PLOT_PARAMS)
    fig.write_html(str(log_dir / 'K-inverse-column-sums.html'))
    fig.write_image(str(log_dir / 'K-inverse-column-sums.pdf'))
