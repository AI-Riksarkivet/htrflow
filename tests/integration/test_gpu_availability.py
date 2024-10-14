import pytest
import torch
from hera.workflows import DAG, Task, Workflow, script, WorkflowsService


SELECTOR_ARGO_SERVER_URL = "http://localhost:2746"
SELECTOR_SERVICE_ACCOUNT = "htrflow-service-account"


@pytest.mark.gpu
def test_gpu_availability():
    assert torch.cuda.is_available(), "CUDA GPU is not available"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")


@script(image="python:3.12")
def echo(message):
    print(message)


with Workflow(
    generate_name="dag-diamond-",
    service_account_name=SELECTOR_SERVICE_ACCOUNT,
    workflows_service=WorkflowsService(host=SELECTOR_ARGO_SERVER_URL),
    entrypoint="diamond",
) as w:
    with DAG(name="diamond"):
        A = Task(name="A", template=echo, arguments={"message": "A"})
        B = Task(name="B", template=echo, arguments={"message": "B"})
        A >> B

w.submit()
