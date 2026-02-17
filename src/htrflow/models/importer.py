import logging

from htrflow.models.huggingface.dit import DiT
from htrflow.models.huggingface.donut import Donut
from htrflow.models.huggingface.trocr import TrOCR, WordLevelTrOCR
from htrflow.models.ultralytics.yolo import YOLO


logger = logging.getLogger(__name__)


def all_models():
    """Import all available models

    Returns a list of all implemented and installed model classes.
    """

    models = [DiT, TrOCR, WordLevelTrOCR, YOLO, Donut]

    try:
        from htrflow.models.teklia.pylaia import PyLaia

        models.append(PyLaia)
    except ModuleNotFoundError:
        pass

    return models
