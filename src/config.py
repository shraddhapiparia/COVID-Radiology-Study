from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], keys: str, default=None):
    cur = d
    for k in keys.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass(frozen=True)
class Paths:
    out_dir: Path
    run_dir: Path
    figures_dir: Path
    models_dir: Path
    tables_dir: Path
    logs_dir: Path


def make_run_dirs(cfg: Dict[str, Any]) -> Paths:
    out_dir = Path(deep_get(cfg, "project.output_dir", "outputs"))
    run_name = deep_get(cfg, "project.run_name", "run")
    run_dir = out_dir / run_name

    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"

    for p in [run_dir, figures_dir, models_dir, tables_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        out_dir=out_dir,
        run_dir=run_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
    )
