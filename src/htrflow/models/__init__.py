from htrflow.models.huggingface.dit import DiT
from htrflow.models.huggingface.donut import Donut
from htrflow.models.huggingface.ppdoclayoutv3 import PPDocLayoutV3
from htrflow.models.huggingface.trocr import TrOCR, WordLevelTrOCR
from htrflow.models.ultralytics.yolo import YOLO


def get_model_by_name(name: str):

    models = available_models()
    for model in models:
        if model.__name__.lower() == name.lower():
            return model

    available = ", ".join(model.__name__ for model in models)
    raise NotImplementedError(f"Model {name} is not supported. The available models are: {available}.")


def available_models():
    """
    Returns a list of all implemented and installed model classes.
    """

    models = [DiT, TrOCR, WordLevelTrOCR, YOLO, Donut, PPDocLayoutV3]

    try:
        from htrflow.models.teklia.pylaia import PyLaia

        models.append(PyLaia)
    except ModuleNotFoundError:
        pass

    return models
