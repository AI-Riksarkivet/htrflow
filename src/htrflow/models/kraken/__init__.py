import importlib.util


# Check that kraken is installed without importing it. Importing kraken also
# loads lightning and scikit-image, which is too slow to do on every `htrflow`
# command, so the classes below import kraken only when a model is created.
# `available_models` relies on this raising when kraken is not installed.
if importlib.util.find_spec("kraken") is None:  # pragma: no cover
    raise ModuleNotFoundError("No module named 'kraken'")

from htrflow.models.kraken.recognition import KrakenRecognition
from htrflow.models.kraken.segmentation import KrakenSegmentation
