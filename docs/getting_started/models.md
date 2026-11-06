
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

## Kraken Models

[Kraken](https://kraken.re) is supported for both segmentation and text
recognition. Install the optional dependency:

```bash
pip install htrflow[kraken]
```

Kraken differs from HTRflow's other segmenters in two ways that affect how you
write a pipeline:

1. **It finds regions and lines in a single pass** and assigns each line to its
   region, so one `Segmentation` step produces the complete region/line tree.
   You do not need the two-stage region-then-lines setup that YOLO requires.
2. **It returns lines already in reading order.** Do *not* add
   `ReadingOrderMarginalia` or `OrderLines` after it — both would re-sort the
   lines and discard Kraken's ordering.

Text recognition uses its own step, `KrakenTextRecognition`, rather than the
generic `TextRecognition`. Kraken recognizers are trained on line images that
Kraken extracts itself from the page, straightening curved lines against their
baselines; the step hands Kraken the whole page so that this still happens.
Passing pre-cropped line images instead costs accuracy on curved writing.

```yaml
steps:
- step: Segmentation
  settings:
    model: KrakenSegmentation
    model_settings:
      model: my-segmentation-model.mlmodel
    generation_settings:
      postprocess: true

- step: KrakenTextRecognition
  settings:
    model: KrakenRecognition
    model_settings:
      model: my-recognition-model.safetensors
```

Lines are recognized in batches of 32 by default. Kraken pads every line in a
batch to the width of the widest one, and recognizers trained on padded lines
are more accurate that way than one line at a time, so keep batches large
unless memory runs short. To change it, set `batch_size` under
`generation_settings`.

Some recognizers are trained on lines padded to a fixed width: Kraken's
PP-OCRv6 trainer pads every line to 2,560 pixels, for example. At inference,
Kraken pads a batch only to its widest line, so such a model sees less padding
than it was trained with. Set `pad_to_width` to the training width to match it:

```yaml
    generation_settings:
      pad_to_width: 2560
```

The width is in pixels of the scaled line images the network sees. Measured
with three PP-OCRv6 models on 503 handwritten lines, it lowered the character
error rate by 0.16 to 0.35 points and reproduced the models' own evaluation.
Padding every batch to full width costs some speed.

### Specifying models

The `model` setting accepts four forms:

| Value | Resolves to |
| --- | --- |
| `my-model.mlmodel` | a local `.mlmodel` or `.safetensors` file, or a directory containing one |
| `10.5281/zenodo.21788409` | a model from the [htrmopo](https://htrmopo.inria.fr) repository, by DOI |
| `Owner/repo` | a Hugging Face repo containing a Kraken model file |
| omitted | the `blla` segmentation model bundled with Kraken (segmentation only) |

Both `.mlmodel` and `.safetensors` weights work, including older models trained
with Kraken 5.

### Segmentation post-processing

Fine-tuned baseline segmenters tend to split single lines into fragments and to
emit short spurious baselines on ascenders. `postprocess` merges the fragments,
drops the spurious lines, and re-derives the line polygons. It is off by
default, so that you get Kraken's own output unless you ask for it. Every
parameter can be set from the YAML:

```yaml
    generation_settings:
      postprocess: true
      postprocess_settings:
        min_len_frac: 0.12     # drop baselines shorter than 0.12 x the page's median baseline
        merge_gap: 1.5         # max gap between two fragments to merge, in line heights
        dy_tol: 0.6            # max height difference where they meet, in line heights
        max_overlap: 0.3       # max horizontal overlap, as a share of the shorter fragment
        scale: 1800            # height the page is resized to while deriving polygons
        repolygonize: true     # derive new polygons from the corrected baselines
        clip_to_region: false  # trim line polygons to their region
        restore_reading_order: true  # re-sort the lines after merging
        topline: false         # baselines under the letters; true if along their tops, null for centerlines
```

`clip_to_region` is worth knowing about: the last line on a page has nothing
below it to bound its polygon, so Kraken tends to extend it into empty space,
which hands the recognizer mostly background. Clipping fixes that, but cuts
real ink wherever a region is drawn tighter than its lines.

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

