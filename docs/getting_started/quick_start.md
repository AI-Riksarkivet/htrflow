# Quickstart (WIP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing) - Quickstart

## Data

Load dataset from huggingface

```python
from datasets import load_dataset

dataset = load_dataset("Riksarkivet/Trolldomkomission")["train"]

images = dataset["image"]
```

## Volume

```python
from htrflow.volume import Volume

vol = Volume([images])

```

## **Segment Images**

```python
from htrflow.models.ultralytics.yolo import YOLO

seg_model = YOLO('ultralyticsplus/yolov8s')
res = seg_model(vol.images()) # vol.segments() is also possible since it points to the images
```

## Update Volume

```python
vol.update(res)
```

## HTR

```python
from htrflow.models.huggingface.trocr import TrOCR

rec_model = TrOCR()
res = rec_model(vol.segments())

vol.update(res)
```

!!! Note

    The final volume
    ```python
        print(vol)
    ```

## Serialize

Saves at outputs/.xml, since the two demo images are called the same, we get only one output file

```python
vol.save('outputs', 'alto')
```

!!! Tip ".."

    Whenever you have large documents, you typicall ...

    ```python
    # Something here
    ```
