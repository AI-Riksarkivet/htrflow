<img src="https://github.com/AI-Riksarkivet/htrflow/blob/main/docs/assets/riks.png?raw=true" width="10%" height="10%" align="right" />

# **htrflow**

<p align="center">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/AI-Riksarkivet/htrflow">
    <img alt="License" src="https://img.shields.io/github/license/AI-Riksarkivet/htrflow">
    <a href="https://circleci.com/gh/AI-Riksarkivet/htrflow">
        <img alt="Build" src="https://img.shields.io/github/AI-Riksarkivet/htrflow/main">
    </a>
    <a href="https://github.com/AI-Riksarkivet/htrflow/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/AI-Riksarkivet/htrflow.svg">
    </a>
    <a href="https://github.com/AI-Riksarkivet/htrflow/releases">
        <img alt="GitHub docs" src="https://img.shields.io/github/docs/AI-Riksarkivet/htrflow.svg">
    </a>
<!-- Add test, ci, build, publish, draft bagdges here... -->

</p>

<p align="center">
  <img src="https://github.com/AI-Riksarkivet/htrflow/blob/main/docs/assets/background_htrflow_2.png?raw=true" alt="HTRFLOW Image" width=70%>
</p>



HTRFlow is an open source tool for handwritten text recognition. It is developed by the AI lab at the Swedish National Archives (Riksarkivet).

## Docs
> ⚠️! Docs still under development

[![mkdocs](https://img.shields.io/badge/htrflow_docs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://ai-riksarkivet.github.io/htrflow)


## Installation

### Package

<a href="https://pypi.org/project/htrflow/">
    <img alt="pypi" src="https://img.shields.io/pypi/pyversions/htrflow">
</a>

Either install with uv or pip:

#### uv:
```sh
uv pip install htrflow[all]
```

#### pip:

```sh
pip install htrflow[all] 
```
This includes ultralytics and transformers models, but if you want also to use openmmlab models:

```sh
pip install htrflow"[all,openmmlab]" 
```

> Note that this forces torch to 2.0.0, since openmmlabs depends on it for now..


### From source
Requirements:
  - [uv](https://docs.astral.sh/uv/) or pip
  - Python 3.10
  - With GPU: CUDA >=11.8 (can still run on CPU)

Clone this repository and run
```sh
uv pip install -e .[all]
```
This will install the HTRFlow CLI and enable huggingface and ultralytics models in a virtual environment. If you also want to use openmmlab models such as RTMDet and Satrn, you also need to run:
```
uv pip install -e ."[all,openmmlab]"
```
Now activate the virtual enviroment with
```
source .venv/bin/activate
```
Or if you are using `uv` you can just run with `uv run` prefix before all your commands:
```
uv run <cmd>
```

The HTRFlow CLI is now available within the `.venv` from uv or where you installed the package. Try it by running:
```sh
htrflow pipeline examples/pipelines/demo.yaml examples/images/pages
```
This command runs HTRFlow on the three example pages in [examples/images/pages](https://github.com/AI-Riksarkivet/htrflow/tree/main/examples/images/pages) and writes the output Page XML and Alto XML.

## Usage
Once HTRFlow is installed, run it with:
```sh
htrflow pipeline <pipeline file> <input image(s)>
```

## Pipelines

HTRFlow is configured with a pipeline file which describes what steps it should perform and which models it should use. Here is an example of a simple pipeline:
```yaml
steps:
- step: Segmentation
  settings:
    model: RTMDet
    model_settings:
       model: Riksarkivet/rtmdet_lines
- step: TextRecognition
  settings:
    model: TrOCR
    model_settings:
       model: Riksarkivet/trocr-base-handwritten-swe
    generation_settings:
       num_beams: 1
- step: RemoveLowTextConfidenceLines
  settings:
    threshold: 0.9
- step: Export
  settings:
    dest: outputs/alto
    format: alto
```
This pipeline uses [Riksarkivet/rtmdet_lines](https://huggingface.co/Riksarkivet/rtmdet_lines) to detect the pages' text lines, then runs [Riksarkivet/trocr-base-handwritten-swe](https://huggingface.co/Riksarkivet/trocr-base-handwritten-swe) to transcribe them, filters the text lines on their confidence score, and then exports the result to Alto XML.

See the demo pipeline  [examples/pipelines/demo.yaml](https://github.com/AI-Riksarkivet/htrflow/tree/main/examples/pipelines/demo.yaml) for a more complex pipeline.

### Built-in pipeline steps
HTRflow comes with several pre-defined pipeline steps out of the box. These include:
- Inference, including text recognition and segmentation
- Image preprocessing
- Reading order detection
- Filtering
- Export


### Custom pipeline steps
You can define your own custom pipeline step by subclassing `PipelineStep` and defining the `run()` method. It takes a `Collection` and returns a `Collection`:
```python
class MyPipelineStep(PipelineStep):
    """A custom pipeline step"""
    def run(self, collection: Collection) -> Collection:
        for page in collection:
            # Do something
        return collection
```
You can add parameters to your pipeline step by also defining the `__init__()` method. It can take any number of arguments. Here, we add one argument, which can be accessed when the step is run:
```python
class MyPipelineStep(PipelineStep):
    """A custom pipeline step"""
    def __init__(self, arg):
        self.arg = arg

    def run(self, collection: Collection) -> Collection:
        for page in collection:
            # Do something
            if self.arg:
              ...
        return collection
```

To use the pipeline step in a pipeline, add the following to your pipeline file:
```yaml
steps:
  - step: MyPipelineStep
    settings: 
      arg: value
```
All key-value pairs listed under `settings` will be passed to the step's `__init__()` method. If the pipeline step doesn't need any arguments, you can omit `settings`.

For filtering and image processing operations, you can base your custom step on the base classes Prune and ProcessImages. Examples of this, and other pipeline steps, can be found in [htrflow/pipeline/steps.py](https://github.com/AI-Riksarkivet/htrflow/blob/56d70ad9e6d8aa38893ae3f46fa90f40311de195/src/htrflow/pipeline/steps.py).


## Models

The following model architectures are currently supported by HTRFlow:
| Model     | Type      | Fine-tuned by the AI lab|
| ------------- | ------------- | --- |
| TrOCR | Text recognition | [Riksarkivet/trocr-base-handwritten-swe](https://huggingface.co/Riksarkivet/trocr-base-handwritten-swe) |
| Satrn | Text recognition | [Riksarkivet/satrn_htr](https://huggingface.co/Riksarkivet/satrn_htr) |
| RTMDet | Segmentation | [Riksarkivet/rtmdet_lines](https://huggingface.co/Riksarkivet/rtmdet_lines) <br> [Riksarkivet/rtmdet_regions](https://huggingface.co/Riksarkivet/rtmdet_regions) |
| Yolo | Segmentation |  |
| DiT | Image classification |  |
