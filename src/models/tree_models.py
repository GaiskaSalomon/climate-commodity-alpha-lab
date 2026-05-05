from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor


def make_xgb_regressor(**kwargs) -> XGBRegressor:
    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def make_lgbm_regressor(**kwargs) -> LGBMRegressor:
    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


def make_rf_regressor(**kwargs) -> RandomForestRegressor:
    defaults = dict(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)
