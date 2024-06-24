---
icon: octicons/home
hide:
  - navigation
---

<img src="assets/riks.png" width="20%" height="20%" align="right" />

# **htrflow_core**

<p align="center">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Swedish-National-Archives-AI-lab/htrflow_core">
    <img alt="License" src="https://img.shields.io/github/license/Swedish-National-Archives-AI-lab/htrflow_core">
    <a href="https://circleci.com/gh/Swedish-National-Archives-AI-lab/htrflow_core">
        <img alt="Build" src="https://img.shields.io/github/Swedish-National-Archives-AI-lab/htrflow_core/main">
    </a>
    <a href="https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Swedish-National-Archives-AI-lab/htrflow_core.svg">
    </a>
    <a href="https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/releases">
        <img alt="GitHub docs" src="https://img.shields.io/github/docs/Swedish-National-Archives-AI-lab/htrflow_core.svg">
    </a>
</p>




htrflow_core is a part of the htrflow suite, which is Riksarkivets open source project for handwritten text recogntion.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.



## Why should I use htrflow?

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.

!!! info inline end "Release Notes"

    [Releases on Github :fontawesome-brands-github:](https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/releases){ .md-button }

 

## Something here..

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.

## Usage

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.

```python
from datasets import load_dataset

dataset = load_dataset("Riksarkivet/Trolldomkomission")["train"]

images = dataset["image"]
```


###  Installation

=== "Installation"

    You can install `htrflow_core` with [pypi](https://pypi.org/project/htrflow_core) in a
    [**Python>=3.10**](https://www.python.org/) environment.

    !!! tip "pip install (recommended)"

        === "core"
            The core installation of `htrflow_core` install everything you need to get you started with structuring output in your htr workflow.

            ```bash
            pip install htrflow_core
            ```

        === "models"
            This installation add support för models we have implemented.

            ```bash
            pip install "htrflow_core[models]"
            ```

=== "Development"

    !!! example "git clone (for development)"

        === "virtualenv"

            ```bash
            # clone repository and navigate to root directory
            git clone https://github.com/Swedish-National-Archives-AI-lab/htrflow_core
            cd htrflow_core

            # setup python environment and activate it
            python3 -m venv venv
            source venv/bin/activate
            pip install --upgrade pip

            # core install
            pip install -e "."

            # all models install
            pip install -e ".[huggingface, openmmlab, ultralytics]"
            ```

        === "poetry"

            ```bash
            # clone repository and navigate to root directory
            git clone https://github.com/Swedish-National-Archives-AI-lab/htrflow_core
            cd htrflow_core

            # setup python environment and activate it
            poetry env use python3.10
            poetry shell

            # core install
            poetry install

            # all models install
            poetry install --all-extras

            # or specific framework
            poetry install --extras huggingface
            ```
