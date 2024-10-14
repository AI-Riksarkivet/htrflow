
# Models

## Supported Models

HTRflow natively supports specific models from the following frameworks: Ultralytics, Hugging Face, and OpenMMLab.

!!! tip
    For a complete list of predefined models compatible with our [Pipeline steps](../reference/pipeline-steps.md), see the [Model reference](../reference/models.md).

## Riksarkivet Models

 Riksarkivet provides several ready-to-use models available on [Hugging Face](https://huggingface.co/Riksarkivet).

Here are some of the different materials the models were trained on:

<iframe
  src="https://huggingface.co/datasets/Riksarkivet/test_images_demo/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## OpenMMLab Models

To use OpenMMLab models (e.g SATRN), specific dependencies need to be installed, including `torch`, `mmcv`, `mmdet`, `mmengine`, `mmocr`, and `yapf`. Follow the instructions below to ensure the correct versions are installed.

!!! Note
    OpenMMLab requires a specific PyTorch version. Make sure you have `pytorch==2.0.0` installed:

    ```bash
    pip install -U torch==2.0.0
    ```

You can install the OpenMMLab dependencies using either `mim` or `pip`.

=== "Using mim"

    The recommended method, according to OpenMMLab, is to use `mim`, which is a package and model manager.

    First, install `mim`:

    ```bash
    pip install -U openmim
    ```

    Then, use `mim` to install the required packages:

    ```bash
    mim install -U mmdet
    mim install -U mmengine
    mim install -U mmocr
    mim install -U mmcv
    ```

=== "Using pip"

    Alternatively, you can install the dependencies using `pip`:

    ```bash
    pip install -U mmcv==2.0.0
    pip install -U mmdet==3.1.0
    pip install -U mmengine==0.7.2
    pip install -U mmocr==1.0.1
    pip install -U yapf==0.40.1
    ```

Here are links to the documentation for each OpenMMLab package used in HTRflow:

- [mim](https://openmim.readthedocs.io/en/latest/)
- [mmdet](https://mmdetection.readthedocs.io/en/latest/overview.html)
- [mmocr](https://mmocr.readthedocs.io/en/latest/get_started/overview.html)
- [mmengine](https://mmengine.readthedocs.io/en/latest/)
- [mmcv](https://mmcv.readthedocs.io/en/latest/)

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

