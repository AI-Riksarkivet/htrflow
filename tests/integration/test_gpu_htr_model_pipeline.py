import os
import pytest
from typer.testing import CliRunner
from htrflow.cli import app

runner = CliRunner()

image_path = "tests/integration/data/trocr_example.png"
pipeline_path = "tests/integration/data/test_gpu_htr_model_pipeline.yaml"


@pytest.fixture(scope="module")
def check_test_files():
    assert os.path.exists(image_path), f"Test image not found: {image_path}"
    assert os.path.exists(
        pipeline_path
    ), f"Test pipeline YAML not found: {pipeline_path}"


@pytest.mark.gpu
def test_run_htr_pipeline(check_test_files):
    result = runner.invoke(
        app,
        [
            "pipeline",
            pipeline_path,
            image_path,
            "--batch-output",
            "1",
            "--logfile",
            "tox-test.log",
        ],
    )

    assert (
        result.exit_code == 0
    ), f"Pipeline returns sucessfully exit code {result.exit_code}"
