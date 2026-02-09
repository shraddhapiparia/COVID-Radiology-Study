from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from .config import deep_get


def report_missingness(df: pd.DataFrame, paths, top_k: int = 30) -> None:
    miss = df.isna().mean().sort_values(ascending=False).head(top_k)
    out = paths.logs_dir / "missingness_top.csv"
    miss.to_csv(out, header=["missing_frac"])
    (paths.logs_dir / "data_shape.txt").write_text(f"rows={len(df)} cols={df.shape[1]}\n")


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def build_new_finding_from_text(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    enabled = deep_get(cfg, "labels.new_finding_from_text.enabled", False)
    if not enabled:
        return df

    text_col = deep_get(cfg, "data.text_col")
    target_col = deep_get(cfg, "data.target_col", "new_finding")

    if not text_col or text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' missing; cannot compute {target_col}.")

    normal_terms = deep_get(cfg, "labels.new_finding_from_text.normal_terms", [])
    stable_terms = deep_get(cfg, "labels.new_finding_from_text.stable_terms", [])
    default_val = int(deep_get(cfg, "labels.new_finding_from_text.default_new_finding_value", 1))

    n_terms = [_normalize_text(t) for t in normal_terms]
    s_terms = [_normalize_text(t) for t in stable_terms]

    def classify(text):
        t = _normalize_text(text)
        if any(term in t for term in s_terms):
            return 0
        if any(term in t for term in n_terms):
            return 0
        return default_val

    df[target_col] = df[text_col].fillna("").map(classify).astype(int)
    return df


def apply_variant_proxy(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    vp = deep_get(cfg, "features.variant_proxy") or deep_get(cfg, "variant_proxy")
    if not isinstance(vp, dict) or not vp.get("enabled", False):
        return df

    date_col = deep_get(cfg, "data.date_col")
    if not date_col or date_col not in df.columns:
        return df

    out_col = vp.get("output_col", "variant_proxy")
    unknown = vp.get("unknown_label", "other")

    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")

    def in_range(a, b):
        a = pd.to_datetime(a)
        b = pd.to_datetime(b)
        return (dt >= a) & (dt <= b)

    labels = pd.Series([unknown] * len(df), index=df.index)

    for k, rng in vp.items():
        if k in {"enabled", "output_col", "unknown_label", "label_as_proxy", "note"}:
            continue
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            mask = in_range(rng[0], rng[1])
            labels.loc[mask] = k

    df[out_col] = labels
    return df


def drop_or_keep(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    strat = deep_get(cfg, "missingness.strategy", "impute")
    if strat == "drop":
        # drop rows with missing target or id (critical)
        target = deep_get(cfg, "data.target_col")
        id_col = deep_get(cfg, "splits.group_col") or deep_get(cfg, "data.id_col")
        keep = df.dropna(subset=[c for c in [target, id_col] if c and c in df.columns])
        return keep
    return df 
