from pathlib import Path
from typing import Mapping
import logging
import numpy as np
import torch
from torch.distributions import Normal
import plotly.graph_objects as go

from .gp import GraphGP, ExpGraphGP, SqGraphGP
# from .generate_test_graphs import random_sampling
from utils.nas import index_to_nx, random_sampling
from utils.plotting import PLOT_PARAMS

logger = logging.getLogger(__name__)

class IntegrandModel(torch.nn.Module):
    """Integrand Model that provides methods to do quadrautre."""

    def __init__(
        self,
        surrogate: GraphGP = None,
        prior=None
    ):
        """Initialise the integrand model.
        
        :param surrogate: GP surrogate.
        :param prior: The prior over graphs. Currently restricted to
            uniform.
        """
        super().__init__()
        if prior is not None:
            raise NotImplementedError('Prior must be uniform')
        
        self.surrogate = surrogate
        self.prior = 1 / 15625
        self._kernel_integrals = None
        self._quadrature_weights = None
        self._log_model_evidence = None

    def compute_kernel_sums(
        self,
        num_samples: int = 2048,
        batch_size: int = None,
        api = None,
        dataset: str = None,
        train_epochs: str = "200",
        restrict_archs_to_best: int = None,
        exact_sum: bool = False,
        log_dir: Path = None,
        smoke_test: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Compute the quadrature weights.

        :param num_samples: Number of monte carlo samples to use for the
            kernel integrals.
        :param batch_size: Number of batches to use for MC for the
            kernel integrals.
        :param api: The NATS-Bench API.
        :param dataset: The dataset.
        :param train_epochs: The number of training epochs.
        :param restrict_archs_to_best: Restrict the prior over
            architectures to the best (as measured by validation loss)
            M.
        :param exact_sum: Whether to compute the kernel integrals
            exactly. If True num_samples must match the prior size.
        :param log_dir: If provided, the location to which to save a
            plot of the kernel integral.
        :param smoke_test: Whether to run with quickest settings for
            debugging.
        :return: Bayesian quadrature weights.
        """
        batch_size = num_samples if batch_size is None else batch_size
        samples_remaining = 10 if smoke_test else num_samples
        int_kernel = torch.zeros(self.surrogate.K_i.size(0))
        samples_used = 0
        counter = 0
        while samples_remaining > 0:
            logger.info(f'Kernel Integral Sample Batch {counter}')
            pool_size = (
                batch_size if samples_remaining > batch_size
                else samples_remaining
            )
            # Sample graphs.
            if exact_sum:
                # Using this order ensures that invalid archiectectures all
                # end up in the last batch.
                pickle = np.load(
                    f'{log_dir.parent}/sandbox/nas/{dataset}/ranked_architectures.npz'
                )
                sample_indices = pickle['ranking'].tolist()[
                    counter * batch_size:(counter + 1) * batch_size
                ]
                samples = [
                    index_to_nx(api, i, hp=train_epochs)
                    for i in sample_indices
                ]
                samples = [s for s in samples if s is not None]
            else:
                log_dir_parent = log_dir.parent if log_dir is not None else Path('../logs')
                samples = random_sampling(
                    pool_size,
                    api,
                    dataset,
                    hp=train_epochs,
                    restrict_to_best=restrict_archs_to_best,
                    log_dir=log_dir_parent
                )
                samples = [s for s in samples if s is not None]
                pool_size = len(samples)
            # Compute the kernel between the data and the samples, [N, M]. Then
            # sum over samples.
            try:
                int_kernel += self.surrogate.cross_covariance(samples).sum(dim=1)
                samples_used += len(samples)
            except Exception as e:
                logger.warning(f'Batch {counter} failed.')
                num_samples -= pool_size
            samples_remaining -= pool_size
            counter += 1
        int_kernel = int_kernel / samples_used
        self._kernel_integrals = int_kernel
        if log_dir is not None:
            np.save(str(log_dir / 'kernel-integral.npy'), int_kernel.numpy())
            fig = go.Figure(
                data=go.Bar(
                    x=np.arange(int_kernel.shape[0]), y=int_kernel.numpy()
                )
            )
            fig.update_layout(**PLOT_PARAMS)
            fig.write_html(str(log_dir / 'kernel-integral.html'))
            fig.write_image(str(log_dir / 'kernel-integral.pdf'))
        return int_kernel
    
    def compute_quadrature_weights(
        self,
        num_samples: int = 2048,
        batch_size: int = None,
        api = None,
        dataset: str = None,
        train_epochs: str = "200",
        restrict_archs_to_best: int = None,
        exact_sum: bool = False,
        log_dir: Path = None,
        smoke_test: bool = False
    ) -> torch.Tensor:
        """Compute the quadrature weights.

        :param num_samples: Number of monte carlo samples to use for the
            kernel integrals.
        :param batch_size: Number of batches to use for MC for the
            kernel integrals.
        :param api: The NATS-Bench API.
        :param dataset: The dataset.
        :param train_epochs: The number of training epochs.
        :param restrict_archs_to_best: Restrict the prior over
            architectures to the best (as measured by validation loss)
            M.
        :param exact_sum: Whether to compute the kernel integrals
            exactly. If True num_samples must match the prior size.
        :param log_dir: If provided, the location to which to save a
            plot of the kernel integral.
        :param smoke_test: Whether to run with quickest settings for
            debugging.
        :return: Bayesian quadrature weights.
        """
        int_kernel = self.compute_kernel_sums(
            num_samples=num_samples,
            batch_size=batch_size,
            api=api,
            dataset=dataset,
            train_epochs=train_epochs,
            restrict_archs_to_best=restrict_archs_to_best,
            exact_sum=exact_sum,
            log_dir=log_dir,
            smoke_test=smoke_test
        )
        quadrature_weights = (int_kernel.view(1, -1) @ self.surrogate.K_i).view(-1)
        self._quadrature_weights = quadrature_weights
        return quadrature_weights

    def posterior(
        self,
        weights: torch.Tensor = None
    ) -> Normal:
        """Compute the posterior over the integral."""
        if weights is None:
            weights = self.compute_quadrature_weights()
        integral_mean = weights @ self.surrogate.y + self.surrogate.y_mean
        self._log_model_evidence = integral_mean.log()
        return Normal(loc=integral_mean, scale=1)


class SqIntegrandModel(IntegrandModel):
    """Integrand Model that provides methods to do sqrt transformed BQ.
    """

    def __init__(
        self,
        surrogate: SqGraphGP = None,
        prior=None
    ):
        """Initialise the integrand model.
        
        :param surrogate: GP surrogate.
        :param prior. The prior over graphs. Currently restricted to
            uniform.
        """
        super().__init__()
        if prior is not None:
            raise NotImplementedError('Prior must be uniform')
        
        self.surrogate = surrogate
        self.prior = 1 / 15625
        self._kernel_integrals = None
        self._quadrature_weights = None
        self._log_model_evidence = None
    
    def compute_quadrature_weights(
        self,
        num_samples: int = 1024,
        batch_size: int = None,
        api = None,
        dataset: str = None,
        train_epochs: str = "200",
        log_dir: Path = None,
        smoke_test: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Compute the quadrature weights.

        :param num_samples: Number of monte carlo samples to use for the
            kernel integrals.
        :param batch_size: Number of batches to use for MC for the
            kernel integrals. Does not make sense for this transform.
        :param api: The NATS-Bench API.
        :param dataset: The dataset.
        :param train_epochs: The number of training epochs.
        :param smoke_test: Whether to run with quickest settings for
            debugging.
        :param log_dir: The logs directory.
        :param kwargs: Swallow keyword arguments.
        :return: Bayesian quadrature weights.
        """
        if batch_size is not None:
            raise NotImplementedError(
                'Batching not implemented for WSABI integrals.'
            )
        num_samples = 10 if smoke_test else num_samples
        logger.info(f'Computing quadrature weights.')
        # Sample graphs.
        samples = random_sampling(
            num_samples,
            api,
            dataset,
            hp=train_epochs,
            log_dir=log_dir.parent
        )
        cross_cov = self.surrogate.cross_covariance(samples)  # [N, M]
        int_kk = cross_cov @ cross_cov.t() / num_samples  # [N, N]
        self._kernel_integrals = int_kk
        quadrature_weights = (
            self.surrogate.K_i @ int_kk @ self.surrogate.K_i  # [N, N]
        )
        self._quadrature_weights = quadrature_weights
        return quadrature_weights
    
    def posterior(
        self,
        weights: torch.Tensor = None
    ) -> Normal:
        """Compute the posterior over the integral."""
        if weights is None:
            weights = self.compute_quadrature_weights()
        int_mean = self.surrogate.y @ weights @ self.surrogate.y
        self._log_model_evidence = int_mean.log()
        return Normal(loc=int_mean, scale=1)
    
    def posterior_predictive(
        self,
        weights: torch.Tensor,
        posteriors: torch.Tensor,
        log_model_evidence: torch.Tensor = None,
        num_samples: int = 2048,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Compute posterior predictive for the model.
        
        :param weights: [N, N].
        :param posteriors: Predictions, should have shape [B, N, C].
        :return predictions: Shape [B, C].
        """
        scaling = self.surrogate.log_offset.exp()
        if log_model_evidence is None:
            log_model_evidence = torch.log(
                self.posterior(weights, num_samples, batch_size).loc * scaling
            )
        posteriors = (
            torch.sqrt(2 * (posteriors - self.surrogate.alpha))
            * self.surrogate.y.unsqueeze(-1)
        )  # [B, N, C]
        integral = (posteriors * (weights @ posteriors)).sum(1)  # [B, C]
        return integral * scaling / log_model_evidence.exp()


class ExpIntegrandModel(IntegrandModel):
    """Integrand Model that provides methods to do log transform BQ."""

    def __init__(
        self,
        surrogate: ExpGraphGP = None,
        prior=None
    ):
        """Initialise the integrand model.
        
        :param surrogate: GP surrogate.
        :param prior: The prior over graphs. Currently restricted to
            uniform.
        """
        super().__init__()
        if prior is not None:
            raise NotImplementedError('Prior must be uniform')
        self.surrogate = surrogate
        self.prior = 1 / 15625
        self._kernel_integrals = None
        self._quadrature_weights = None
        self._log_model_evidence = None
    
    def compute_quadrature_weights(
        self,
        num_samples: int = 2048,
        batch_size: int = None,
        api = None,
        dataset: str = None,
        train_epochs: str = "200",
        restrict_archs_to_best: int = None,
        exact_sum: bool = False,
        log_dir: Path = None,
        smoke_test: bool = False
    ) -> torch.Tensor:
        """Compute the quadrature weights.

        :param num_samples: Number of monte carlo samples to use for the
            kernel integrals.
        :param batch_size: Number of batches to use for MC for the
            kernel integrals. Does not make sense for this transform.
        :param api: The NATS-Bench API.
        :param smoke_test: Whether to run with quickest settings for
            debugging.
        :return: Bayesian quadrature weights.
        """
        if batch_size is not None:
            raise NotImplementedError("Batching doesn't make sense for MMLT.")
        num_samples = 10 if smoke_test else num_samples
        logger.info(f'Computing quadrature weights.')
        # Sample graphs.
        samples = random_sampling(
            num_samples,
            api,
            dataset,
            hp=train_epochs,
            log_dir=log_dir.parent
        )
        architecture_weights = (
            self.surrogate.cross_covariance(samples).t() @ self.surrogate.K_i
        )  # [M, N]
        var_weights = self.surrogate.predict(samples)[1].diag()  # [M]
        sample_weights = (
            architecture_weights @ self.surrogate.y + 0.5 * var_weights
        )  # [M]
        quadrature_weights = {
            'architecture_weights': architecture_weights,
            'sample_weights': sample_weights
        }
        self._quadrature_weights = quadrature_weights
        return quadrature_weights

    def posterior(
        self,
        weights: Mapping[str, torch.Tensor] = None,
        num_samples: int = 2048,
        batch_size: int = None,
    ) -> Normal:
        """Compute the posterior over the integral."""
        if weights is None:
            weights = self.compute_quadrature_weights(num_samples, batch_size)
        exponents = weights['sample_weights']
        num_samples = torch.tensor(exponents.size(0))
        int_mean = (exponents.logsumexp(0) - num_samples.log()).exp()
        self._log_model_evidence = int_mean
        return Normal(loc=int_mean, scale=1)
    
    def posterior_predictive(
        self,
        weights: Mapping[str, torch.Tensor],
        posteriors: torch.Tensor,
        log_model_evidence: torch.Tensor = None,
        num_samples: int = 2048,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Compute posterior predictive for the model.
        
        :param weights: Should have keys 'mean_weights' and
            'var_weights' which correspond to tensors of shape [M, N]
            and [M].
        :param posteriors: Predictions, should have shape [B, N, C].
        :return predictions: Shape [B, C].
        """
        if log_model_evidence is None:
            log_model_evidence = self.posterior(
                weights, num_samples, batch_size
            ).loc.log() + self.surrogate.log_offset

        exponents_Z = weights['sample_weights']  # [M]
        # [B, M, C]
        exponents_p = weights['architecture_weights'] @ posteriors.log()
        combined_posteriors = (
            torch.logsumexp(
                exponents_p + exponents_Z.unsqueeze(0).unsqueeze(-1), 1
            )
            - torch.tensor(exponents_Z.size(0)).log()
            - log_model_evidence
            + self.surrogate.log_offset
        ).exp_()
        return combined_posteriors
