import logging

from htrflow.models.huggingface.dit import DiT
from htrflow.models.huggingface.llava_next import LLavaNext
from htrflow.models.huggingface.trocr import TrOCR, WordLevelTrOCR
from htrflow.models.ultralytics.yolo import YOLO


logger = logging.getLogger(__name__)


def all_models():
    """Import all available models

    Returns a list of all implemented and installed model classes.
    """

    models = [DiT, LLavaNext, TrOCR, WordLevelTrOCR, YOLO]

    openmmlabs_models = []
    teklia_models = []

    # Import openmmlabs models if available
    try:
        from htrflow.models.openmmlab.rtmdet import RTMDet
        from htrflow.models.openmmlab.satrn import Satrn

        openmmlabs_models = [RTMDet, Satrn]

        models += openmmlabs_models

    except ModuleNotFoundError:
        logger.exception(f"Could not import OpenMMLab models: {openmmlabs_models}.")

    try:
        from htrflow.models.teklia.pylaia import PyLaia

        teklia_models = [PyLaia]

        models += teklia_models

    except ModuleNotFoundError:
        logger.exception(f"Could not import Teklia models: {teklia_models}.")

    return models
