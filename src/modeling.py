from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from .config import deep_get


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: Optional[np.ndarray]
    groups_test: Optional[np.ndarray]


def make_preprocessor(cfg: Dict[str, Any], X: pd.DataFrame) -> ColumnTransformer:
    miss_num = deep_get(cfg, "missingness.numeric_imputer", "median")
    miss_cat = deep_get(cfg, "missingness.categorical_imputer", "most_frequent")

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers = []

    if num_cols:
        transformers.append(
            ("num",
             Pipeline([("imputer", SimpleImputer(strategy=miss_num))]),
             num_cols)
        )

    if cat_cols:
        transformers.append(
            ("cat",
             Pipeline([
                 ("imputer", SimpleImputer(strategy=miss_cat)),
                 ("onehot", OneHotEncoder(handle_unknown="ignore"))
             ]),
             cat_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_estimator(cfg: Dict[str, Any]) -> RandomForestClassifier:
    params = deep_get(cfg, "model.params", {})
    params = dict(params) if isinstance(params, dict) else {}
    params.setdefault("n_estimators", 500)
    params.setdefault("random_state", deep_get(cfg, "project.random_seed", 42))
    params.setdefault("n_jobs", -1)
    params.setdefault("class_weight", "balanced")
    return RandomForestClassifier(**params)


def build_pipeline(cfg: Dict[str, Any], X: pd.DataFrame) -> Pipeline:
    pre = make_preprocessor(cfg, X)
    est = make_estimator(cfg)
    return Pipeline([("preprocess", pre), ("model", est)])


def split_data(df: pd.DataFrame, cfg: Dict[str, Any], feature_cols: list[str]) -> SplitData:
    target = deep_get(cfg, "data.target_col")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    group_col = deep_get(cfg, "splits.group_col")
    strategy = deep_get(cfg, "splits.strategy", "group_shuffle")
    test_size = float(deep_get(cfg, "splits.test_size", 0.2))
    rs = int(deep_get(cfg, "splits.random_state", deep_get(cfg, "project.random_seed", 42)))

    X = df[feature_cols].copy()
    y = df[target].astype(int).to_numpy()

    groups = df[group_col].to_numpy() if group_col and group_col in df.columns else None

    if strategy == "group_shuffle":
        if groups is None:
            raise ValueError("group_shuffle requires splits.group_col present in dataframe.")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    elif strategy == "group_kfold":
        if groups is None:
            raise ValueError("group_kfold requires splits.group_col present in dataframe.")
        # For holdout, take the first fold as test
        gkf = GroupKFold(n_splits=int(deep_get(cfg, "splits.n_splits", 5)))
        train_idx, test_idx = next(gkf.split(X, y, groups=groups))

    else:
        raise ValueError(f"Unsupported splits.strategy='{strategy}'. Use group_shuffle or group_kfold.")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx] if groups is not None else None
    g_test = groups[test_idx] if groups is not None else None

    return SplitData(X_train, X_test, y_train, y_test, g_train, g_test)


def fit_with_grid_search(cfg: Dict[str, Any], pipe: Pipeline, split: SplitData) -> GridSearchCV | Pipeline:
    gs_cfg = deep_get(cfg, "model.grid_search", {})
    enabled = bool(gs_cfg.get("enabled", False))
    if not enabled:
        pipe.fit(split.X_train, split.y_train)
        return pipe

    scoring = gs_cfg.get("scoring", "f1")
    cv = int(gs_cfg.get("cv", 5))
    param_grid = gs_cfg.get("param_grid", {})

    grid = {f"model__{k}": v for k, v in param_grid.items()}

    # CV splitter
    stratify = bool(deep_get(cfg, "splits.stratify", True))
    if split.groups_train is not None:
        if stratify and HAS_SGK:
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True,
                                            random_state=int(deep_get(cfg, "splits.random_state", 42)))
        else:
            splitter = GroupKFold(n_splits=cv)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=splitter,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(split.X_train, split.y_train, groups=split.groups_train)
        return gs

    # fallback if no groups
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    gs.fit(split.X_train, split.y_train)
    return gs


def choose_threshold(cfg: Dict[str, Any], y_true: np.ndarray, y_prob: np.ndarray) -> float:
    th_cfg = deep_get(cfg, "evaluation.threshold", {})
    if not isinstance(th_cfg, dict) or not th_cfg.get("enabled", False):
        return 0.5
    method = th_cfg.get("method", "f1_opt")

    thresholds = np.linspace(0.05, 0.95, 19)
    best_th, best_val = 0.5, -1.0

    if method == "f1_opt":
        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_val:
                best_val, best_th = f1, th
        return float(best_th)

    return 0.5
