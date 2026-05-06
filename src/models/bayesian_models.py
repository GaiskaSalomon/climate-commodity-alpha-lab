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


class BayesianLinearRegressor:
    """
    Bayesian linear regression with weakly informative priors.

    Wraps PyMC MCMC inference with a sklearn-like interface:
        blr.fit(X, y)
        mean, std = blr.predict(X_new)
        summary = blr.coefficient_summary()

    Priors:
        alpha ~ Normal(0, 0.05)     — intercept
        betas ~ Normal(0, 0.1)      — coefficients (one per feature)
        sigma ~ HalfNormal(0.05)    — observation noise
    """

    def __init__(
        self,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        random_seed: int = 42,
    ):
        if not HAS_PYMC:
            raise ImportError("Install pymc and arviz: pip install pymc arviz")
        self.draws       = draws
        self.tune        = tune
        self.chains      = chains
        self.random_seed = random_seed
        self._idata      = None
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        n_features = X.shape[1]
        self._feature_names = (
            feature_names if feature_names is not None
            else [f"x{i}" for i in range(n_features)]
        )

        with pm.Model():
            alpha = pm.Normal("alpha", mu=0, sigma=0.05)
            betas = pm.Normal("betas", mu=0, sigma=0.1, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=0.05)

            mu = alpha + pm.math.dot(X, betas)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            self._idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=True,
            )
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (posterior_mean, posterior_std) for each row of X."""
        alpha_samples = self._idata.posterior["alpha"].values.flatten()
        betas_samples = self._idata.posterior["betas"].values.reshape(-1, X.shape[1])

        # shape: (n_posterior_samples, n_test_rows)
        preds = alpha_samples[:, None] + betas_samples @ X.T
        return preds.mean(axis=0), preds.std(axis=0)

    def coefficient_summary(self) -> pd.DataFrame:
        """Return a DataFrame with posterior stats for each feature coefficient."""
        betas = self._idata.posterior["betas"].values.reshape(-1, len(self._feature_names))
        mean  = betas.mean(axis=0)
        sd    = betas.std(axis=0)
        hdi   = az.hdi(self._idata, var_names=["betas"], hdi_prob=0.94)["betas"].values
        return pd.DataFrame({
            "feature": self._feature_names,
            "mean":    mean,
            "sd":      sd,
            "hdi_3%":  hdi[:, 0],
            "hdi_97%": hdi[:, 1],
        })

    @property
    def idata(self):
        return self._idata


# ---------------------------------------------------------------------------
# Procedural helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fit_bayesian_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    samples: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
):
    """Bayesian linear regression. Returns an InferenceData object (arviz)."""
    blr = BayesianLinearRegressor(
        draws=samples, tune=tune, chains=chains, random_seed=random_seed
    )
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    blr.fit(
        X_train.values if hasattr(X_train, "values") else X_train,
        y_train.values if hasattr(y_train, "values") else y_train,
        feature_names=feature_names,
    )
    return blr.idata


def predict_bayesian(idata, X_test: pd.DataFrame) -> np.ndarray:
    """Generate posterior predictive mean from fitted Bayesian model."""
    if not HAS_PYMC:
        raise ImportError("pymc and arviz are required.")

    alpha_samples = idata.posterior["alpha"].values.flatten()
    betas_samples = idata.posterior["betas"].values.reshape(-1, X_test.shape[1])
    predictions   = alpha_samples[:, None] + betas_samples @ X_test.values.T
    return predictions.mean(axis=0)
