from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd

from .config import deep_get


def _cols(cfg, key, default=None):
    v = deep_get(cfg, key, default or [])
    return v if isinstance(v, list) else []


def resolve_feature_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    model_set = int(deep_get(cfg, "features.model_set", 1))
    sets = deep_get(cfg, "features.sets", {})
    if model_set not in sets:
        raise ValueError(f"features.model_set={model_set} not present in features.sets")

    include = sets[model_set].get("include", [])

    cols: List[str] = []
    if "radiology_features" in include:
        rad_cols = _cols(cfg, "radiology_features.categories")
        cols += [c for c in rad_cols if c in df.columns]

    if "symptoms" in include:
        cols += [c for c in _cols(cfg, "symptoms.feature_cols") if c in df.columns]

    if "demographics" in include:
        cols += [c for c in _cols(cfg, "demographics.derived_cols") if c in df.columns]
        # Also allow raw demographic cols if derived not present
        cols += [c for c in _cols(cfg, "data.demographics_cols") if c in df.columns and c not in cols]

    if "history" in include:
        cols += [c for c in _cols(cfg, "history.feature_cols") if c in df.columns]

    if "variant_proxy" in include:
        vp_col = deep_get(cfg, "features.variant_proxy.output_col", "variant_proxy")
        if vp_col in df.columns:
            cols.append(vp_col)

    if "variant" in include:
        vcol = deep_get(cfg, "data.variant_col", "variant")
        if vcol in df.columns:
            cols.append(vcol)

    # de-duplicate preserve order
    seen = set()
    final = []
    for c in cols:
        if c not in seen:
            final.append(c)
            seen.add(c)

    if not final:
        raise ValueError("No feature columns resolved. Check YAML and dataframe columns.")
    return final
