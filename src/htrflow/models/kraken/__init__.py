import importlib.util


# Probe without importing: kraken pulls in lightning and scikit-image, which
# is far too slow to pay for on every `htrflow` invocation. `available_models`
# relies on this raising when kraken is not installed.
if importlib.util.find_spec("kraken") is None:  # pragma: no cover
    raise ModuleNotFoundError("No module named 'kraken'")

from htrflow.models.kraken.recognition import KrakenRecognition
from htrflow.models.kraken.segmentation import KrakenSegmentation
