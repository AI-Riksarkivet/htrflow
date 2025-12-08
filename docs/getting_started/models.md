
# Models

Riksarkivet provides several ready-to-use models available on [Hugging Face](https://huggingface.co/Riksarkivet).


## Teklia Models

To use models from Teklia (currently only PyLaia), specific dependencies need to be installed, including `pylaia`.  Follow the instructions below to ensure the correct versions are installed.

```bash
pip install -U pylaia
```

!!! Note
    Pylaia requires a specific PyTorch version. Make sure you have `pytorch==1.13.0` installed:

    ```bash
    pip install -U torch==1.13.0
    ```

!!! Note
    Pylaia requires a specific Python version. Make sure you have `python=<3.10` 

Link to the documentation for PyLaia from Teklia:

- [PyLaia](https://atr.pages.teklia.com/pylaia/get_started/)

## Custom Models

If your model (or framework) is not supported, you can implement a custom model in HTRflow. Below is a basic example:

```python
class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        # Initialize your model here
        pass

    def _predict(self, images, **kwargs) -> list[Result]:
        # Run inference on `images`
        # Should return, for example, Result.text_recognition_result() 
        # or Result.segmentation_result()
```

See [Result reference](../reference/result.md) on different types of return formats from the models. For instance, `Result.text_recognition_result()` for HTR or `Result.segmentation_result()` for segmenetation or object detection.

## Examples of Custom Implementations

**Text Recognition Model:**

```python
class RecognitionModel(BaseModel):
    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        metadata = {"model": "Lorem dummy model"}
        n = 2
        return [
            Result.text_recognition_result(
                metadata,
                texts=[lorem.sentence() for _ in range(n)],
                scores=[random.random() for _ in range(n)],
            )
            for _ in images
        ]
```

**Document Classification Model:**

```python
class ClassificationModel(BaseModel):
    """Model that classifies input images into different types of potato dishes."""

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        classes = ["baked potato", "french fry", "raggmunk"]
        return [
            Result(metadata={"model": "Potato classifier 2000"}, data=[{"classification": random.choice(classes)}])
            for _ in images
        ]
```

