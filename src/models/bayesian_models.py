"""
Bayesian regression module for commodity return forecasting.

Requires: pymc>=5.0, arviz
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


def fit_bayesian_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    samples: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
):
    """
    Bayesian linear regression with weakly informative priors.

    Returns an InferenceData object (arviz).
    """
    if not HAS_PYMC:
        raise ImportError("pymc and arviz are required. Install with: pip install pymc arviz")

    X = X_train.values
    y = y_train.values

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=0.05)
        betas = pm.Normal("betas", mu=0, sigma=0.1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        mu = alpha + pm.math.dot(X, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
        )

    return idata


def predict_bayesian(idata, X_test: pd.DataFrame) -> np.ndarray:
    """
    Generate posterior predictive mean from fitted Bayesian model.
    """
    if not HAS_PYMC:
        raise ImportError("pymc and arviz are required.")

    alpha_samples = idata.posterior["alpha"].values.flatten()
    betas_samples = idata.posterior["betas"].values.reshape(-1, X_test.shape[1])

    predictions = alpha_samples[:, None] + betas_samples @ X_test.values.T
    return predictions.mean(axis=0)
