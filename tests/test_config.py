from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_parses_and_input_exists():
    config_path = REPO_ROOT / "configs" / "demo.yaml"

    assert config_path.exists(), "configs/demo.yaml should exist"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "demo.yaml should parse as a dictionary"
    assert "data" in config, "demo.yaml should contain a data section"
    assert "input_csv" in config["data"], "demo.yaml data section should define input_csv"

    input_csv = REPO_ROOT / config["data"]["input_csv"]
    assert input_csv.exists(), f"Configured input CSV does not exist: {input_csv}"


def test_synthetic_demo_csv_has_required_columns():
    config_path = REPO_ROOT / "configs" / "demo.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    input_csv = REPO_ROOT / config["data"]["input_csv"]
    df = pd.read_csv(input_csv)

    required_columns = {
        config["data"]["id_col"],
        config["data"]["text_col"],
        config["data"]["target_col"],
    }

    covid_status_col = config["data"].get("covid_status_col")
    if covid_status_col:
        required_columns.add(covid_status_col)

    variant_col = config["data"].get("variant_col")
    if variant_col:
        required_columns.add(variant_col)

    missing = required_columns - set(df.columns)
    assert not missing, f"Synthetic demo CSV is missing columns: {missing}"
    assert len(df) > 0, "Synthetic demo CSV should contain at least one row"