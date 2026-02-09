from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from .config import deep_get


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Supports:
      - data.paths (multiple CSVs) -> merged using pat_id 
    """

    paths = deep_get(cfg, "data.paths", {})
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Provide data.paths in YAML.")

    dfs: List[Tuple[str, pd.DataFrame]] = []
    for name, p in paths.items():
        if not p:
            continue
        df = _read_csv(p)
        dfs.append((name, df))

    if not dfs:
        raise ValueError("No valid CSV paths found in data.paths.")

    # Best-effort merge strategy:
    # - If 'pat_id' exists, outer-merge on it
    id_col = deep_get(cfg, "data.id_col") or deep_get(cfg, "data.group_col") or "pat_id"

    if all(id_col in df.columns for _, df in dfs):
        base_name, base = dfs[0]
        for name, df in dfs[1:]:
            # Avoid duplicate columns created by merges; suffix by source name
            overlap = [c for c in df.columns if c in base.columns and c != id_col]
            if overlap:
                df = df.rename(columns={c: f"{c}__{name}" for c in overlap})
            base = base.merge(df, on=id_col, how="outer")
        return base

    raise ValueError(
        f"Could not merge data.paths: '{id_col}' not present in all tables and schemas differ."
    )
