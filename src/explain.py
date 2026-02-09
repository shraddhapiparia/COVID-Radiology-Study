from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd
import shap

from .config import deep_get


def shap_summary(cfg: Dict[str, Any], fitted_estimator, X: pd.DataFrame, out_path, seed: int = 42):
    sh_cfg = deep_get(cfg, "explainability.shap", {})
    if not isinstance(sh_cfg, dict) or not sh_cfg.get("enabled", False):
        return

    max_samples = int(sh_cfg.get("max_samples", 500))
    rs = int(sh_cfg.get("random_state", seed))

    if len(X) > max_samples:
        Xs = X.sample(n=max_samples, random_state=rs)
    else:
        Xs = X

    model = fitted_estimator.best_estimator_ if hasattr(fitted_estimator, "best_estimator_") else fitted_estimator
    # TreeExplainer works on the final RF, but we need transformed features for full correctness.
    # Use SHAP on the model with preprocessed matrix.
    preprocess = model.named_steps["preprocess"]
    rf = model.named_steps["model"]

    Xt = preprocess.transform(Xs)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xt)

    # Feature names from ColumnTransformer/OneHotEncoder (best-effort)
    feature_names = []
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = None

    shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
