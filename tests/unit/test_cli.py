import pytest
import typer
from typer.testing import CliRunner

from htrflow import cli


INPUTS = [f"file{i}" for i in range(10)]


def test_get_inputs_list():
    inputs = cli.get_inputs(INPUTS, None)
    assert inputs == INPUTS


def test_get_inputs_list_and_file():
    with pytest.raises(typer.BadParameter):
        cli.get_inputs(INPUTS, "inputs.txt")


def test_get_inputs_file(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    inputs_file = tmp_path / "inputs.txt"
    inputs_file.write_text("\n".join(INPUTS))
    inputs = list(cli.get_inputs(None, inputs_file))
    assert inputs == INPUTS


def test_pipeline_propagates_worker_exception(monkeypatch):
    from htrflow.pipeline import steps
    from htrflow.pipeline.pipeline import Pipeline

    class FailingPipeline:
        steps = []

        def run(self, document):
            raise RuntimeError("page failed")

    class RecordingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args)

    logger = RecordingLogger()
    monkeypatch.setattr(cli, "setup_pipeline_logging", lambda *args: logger)
    monkeypatch.setattr(Pipeline, "from_config", classmethod(lambda cls, path: FailingPipeline()))
    monkeypatch.setattr(steps, "auto_import", lambda inputs: iter([object()]))

    result = CliRunner().invoke(cli.app, ["pipeline", "pipeline.yaml", "page.jpg"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "page failed"
    assert not any("Processed" in message for message in logger.messages)
