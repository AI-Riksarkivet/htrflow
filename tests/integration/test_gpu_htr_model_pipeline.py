import os

import pytest
from typer.testing import CliRunner

from htrflow.cli import app


runner = CliRunner()


def run_pipeline_test(image_path, pipeline_path):
    """Helper function to run the pipeline test."""
    assert os.path.exists(image_path), f"Test image not found: {image_path}"
    assert os.path.exists(pipeline_path), f"Test pipeline YAML not found: {pipeline_path}"

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

    assert result.exit_code == 0, f"Pipeline returns successfully, exit code {result.exit_code}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "image_path, pipeline_path",
    [
        (
            "tests/integration/data/images/trocr_example.png",
            "tests/integration/data/pipelines/test_gpu_hf_htr_pipeline.yaml",
        ),
        (
            "tests/integration/data/images/trocr_example.png",
            "tests/integration/data/pipelines/test_gpu_svea_htr_pipeline.yaml",
        ),
    ],
)
def test_run_htr_pipeline(image_path, pipeline_path):
    run_pipeline_test(image_path, pipeline_path)


@pytest.mark.openmmlab
@pytest.mark.parametrize(
    "image_path, pipeline_path",
    [
        (
            "tests/integration/data/images/trocr_example.png",
            "tests/integration/data/pipelines/test_gpu_hf_htr_pipeline.yaml",
        ),
        (
            "tests/integration/data/images/trocr_example.png",
            "tests/integration/data/pipelines/test_gpu_opennmlab_htr_pipeline.yaml",
        ),
    ],
)
def test_run_openmmlab_pipeline(image_path, pipeline_path):
    run_pipeline_test(image_path, pipeline_path)


@pytest.mark.teklia
@pytest.mark.parametrize(
    "image_path, pipeline_path",
    [
        (
            "tests/integration/data/images/rimes_example.jpg",
            "tests/integration/data/pipelines/test_gpu_teklia_htr_pipeline.yaml",
        ),
    ],
)
def test_run_teklia_pipeline(image_path, pipeline_path):
    run_pipeline_test(image_path, pipeline_path)
