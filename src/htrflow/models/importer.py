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

    # Import openmmlabs models if available
    try:
        from htrflow.models.openmmlab.rtmdet import RTMDet
        from htrflow.models.openmmlab.satrn import Satrn

        models += [RTMDet, Satrn]

    except ModuleNotFoundError:
        logger.exception("Could not import OpenMMLab models RTMDet and Satrn.")

    return models
