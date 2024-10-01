---
# icon: octicons/archive
hide:
  - navigation
---

<h1 class="hide-title">Htrflow</h1>

<div align="center"">
<img src="assets/background_htrflow_2.png"/>
</div>


<p align="center">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/AI-Riksarkivet/htrflow">
    <img alt="License" src="https://img.shields.io/github/license/AI-Riksarkivet/htrflow">
    <a href="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/publish-pypi.yml">
        <img alt="PyPI" src="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/publish-pypi.yml/badge.svg">
    </a>
    <a href="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/deploy-docs.yml">
        <img alt="Deploy Docs" src="https://github.com/AI-Riksarkivet/htrflow/actions/workflows/deploy-docs.yml/badge.svg">
    </a>
</p>

___


## HTRflow

<img src="assets/lab.png" width="20%" height="20%" align="right" />

HTRflow is an open source tool for HTR and OCR developed by the AI lab at the National Archives of Sweden (*Riksarkivet*).

## Key features

- **Flexibility**: Customize the HTR/OCR process for different kinds of materials.
- **Compatibility**: HTRflow supports all models trained by the AI lab - and more!
- **YAML pipelines**: HTRflow YAML pipelines are easy to create, modify and share.
- **Export**: Export results as Alto XML, Page XML, plain text or JSON.
- **Evaluation**: Compare results from different pipelines with ground truth.

<div align="center"">
<img src="assets/pipeline.png" width="90%"/>
</div>


## Installation
Install HTRflow with pip:
```
pip install htrflow
```
For more details, see the [Installation guide](getting_started/installation.md).

## Getting Started

<img src="assets/worker.png" width="30%" height="30%" align="right" />


Ready to build your own pipeline for your documents? Head over to the [Quickstart guide](getting_started/quick_start.md) to get started with HTRflow.

The guide will walk you through setting up your first pipeline, utilizing pre-trained models, and seamlessly running HTR/OCR tasks. With the HTRflow CLI, you can quickly set up pipelines using `pipeline.yaml` files as your "blueprints".


