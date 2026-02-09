from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np

from .config import load_yaml, make_run_dirs, deep_get
from .io_data import load_dataset
from .preprocess import report_missingness, build_new_finding_from_text, apply_variant_proxy, drop_or_keep
from .features import resolve_feature_columns
from .modeling import build_pipeline, split_data, fit_with_grid_search, choose_threshold
from .evaluate import compute_metrics, plot_roc
from .explain import shap_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = make_run_dirs(cfg)

    # Save config copy
    (paths.run_dir / "config_used.yaml").write_text(open(args.config, "r", encoding="utf-8").read())

    # Load + preprocess
    df = load_dataset(cfg)
    report_missingness(df, paths)

    df = build_new_finding_from_text(df, cfg)
    df = apply_variant_proxy(df, cfg)
    df = drop_or_keep(df, cfg)

    # Resolve features
    feature_cols = resolve_feature_columns(df, cfg)

    # Split
    split = split_data(df, cfg, feature_cols)

    # Model
    pipe = build_pipeline(cfg, split.X_train)
    fitted = fit_with_grid_search(cfg, pipe, split)

    # Predict
    model = fitted.best_estimator_ if hasattr(fitted, "best_estimator_") else fitted
    y_prob = model.predict_proba(split.X_test)[:, 1]
    th = choose_threshold(cfg, split.y_train, (model.predict_proba(split.X_train)[:, 1]))
    y_pred = (y_prob >= th).astype(int)

    # Metrics
    m = compute_metrics(split.y_test, y_pred, y_prob)
    m["threshold"] = float(th)

    pd.DataFrame([m]).to_csv(paths.tables_dir / "metrics.csv", index=False)
    (paths.run_dir / "metrics.json").write_text(json.dumps(m, indent=2))

    # Predictions table
    pred_df = split.X_test.copy()
    pred_df["y_true"] = split.y_test
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(paths.tables_dir / "predictions.csv", index=False)

    # ROC plot
    plot_roc(split.y_test, y_prob, paths.figures_dir / "roc.png")

    # SHAP
    shap_summary(cfg, fitted, split.X_test, paths.figures_dir / "shap_summary.png",
                 seed=int(deep_get(cfg, "project.random_seed", 42)))

    print("Done.")
    print(f"Run dir: {paths.run_dir}")


if __name__ == "__main__":
    main()
