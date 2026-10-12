import pytest

from htrflow.pipeline import steps


class FailingModel:
    metadata = {}

    def __call__(self, images, **kwargs):
        raise RuntimeError("model failed")


def test_inference_propagates_model_exception(monkeypatch):
    step = steps.Inference(FailingModel(), batch_size=1)
    monkeypatch.setattr(steps, "ImageLoader", lambda document: iter([(object(), object())]))

    with pytest.raises(RuntimeError, match="model failed"):
        step.run(object())

    assert step._thread.is_alive()


def test_inference_thread_survives_failed_batch():
    class FailOnceModel:
        metadata = {}

        def __init__(self):
            self.calls = 0

        def __call__(self, images, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model failed")
            return [f"result {index}" for index, _ in enumerate(images)]

    step = steps.Inference(FailOnceModel(), batch_size=2)
    failed_futures = [step._queue.put(item) for item in ("first", "second")]
    for future in failed_futures:
        with pytest.raises(RuntimeError, match="model failed"):
            future.result(timeout=5)

    successful_futures = [step._queue.put(item) for item in ("third", "fourth")]
    assert [future.result(timeout=5) for future in successful_futures] == ["result 0", "result 1"]
    assert step._thread.is_alive()


def test_inference_rejects_wrong_number_of_outputs():
    class ShortModel:
        metadata = {}

        def __call__(self, images, **kwargs):
            return ["only one result"]

    step = steps.Inference(ShortModel(), batch_size=2)
    futures = [step._queue.put(item) for item in ("first", "second")]

    for future in futures:
        with pytest.raises(ValueError, match="returned 1 outputs for a batch of 2 inputs"):
            future.result(timeout=5)


def test_inference_propagates_output_iteration_exception():
    class FailingGeneratorModel:
        metadata = {}

        def __call__(self, images, **kwargs):
            yield "partial result"
            raise RuntimeError("output iteration failed")

    step = steps.Inference(FailingGeneratorModel(), batch_size=2)
    futures = [step._queue.put(item) for item in ("first", "second")]

    for future in futures:
        with pytest.raises(RuntimeError, match="output iteration failed"):
            future.result(timeout=5)
