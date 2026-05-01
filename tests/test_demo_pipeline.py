import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_pipeline_runs_successfully():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run",
            "--config",
            "configs/demo.yaml",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        "Demo pipeline failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )