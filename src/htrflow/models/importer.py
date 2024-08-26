import logging


logger = logging.getLogger(__name__)


def all_models():
    """Import all available models

    Returns a list of all implemented and installed model classes.
    """
    models = []
    # Import huggingface models if available
    try:
        from htrflow.models.huggingface.dit import DiT
        from htrflow.models.huggingface.llava_next import LLavaNext
        from htrflow.models.huggingface.trocr import TrOCR, WordLevelTrOCR

        hf_models = [DiT, LLavaNext, TrOCR, WordLevelTrOCR]
        models.extend(hf_models)
        logger.info(f"Imported huggingface models: {', '.join(model.__name__ for model in hf_models)}")
    except ModuleNotFoundError:
        logger.info("Huggingface models are available but not installed. Install with poetry --extras huggingface")

    # Import openmmlabs models if available
    try:
        from htrflow.models.openmmlab.rtmdet import RTMDet
        from htrflow.models.openmmlab.satrn import Satrn

        mmlabs_models = [RTMDet, Satrn]
        models.extend(mmlabs_models)
        logger.info(f"Imported openmmlabs models: {', '.join(model.__name__ for model in mmlabs_models)}")
    except ModuleNotFoundError:
        logger.info("Openmmlabs models are available but not installed. Install with poetry --extras openmmlabs")

    # Import ultralytics models if available
    try:
        from htrflow.models.ultralytics.yolo import YOLO

        models.append(YOLO)
        logger.info("Imported ultralytics model: YOLO")
    except ModuleNotFoundError:
        logger.info("Ultralytics models are available but not installed. Install with poetry --extras ultralytics")

    return models
