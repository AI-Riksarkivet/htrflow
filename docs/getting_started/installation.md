# Installation / Setup


###  Installation


<!-- termynal -->

```
> pip install htrflow
---> 100%
Installed
```

=== "Installation"

    You can install `htrflow` with [pypi](https://pypi.org/project/htrflow) in a
    [**Python>=3.10**](https://www.python.org/) environment.

    !!! tip "pip install (recommended)"

        === "core"
            The core installation of `htrflow` install everything you need to get you started with structuring output in your htr workflow.

            ```bash
            pip install htrflow
            ```

        === "models"
            This installation add support för models we have implemented.

            ```bash
            pip install "htrflow[models]"
            ```

=== "Development"

    !!! example "git clone (for development)"

        === "virtualenv"

            ```bash
            # clone repository and navigate to root directory
            git clone https://github.com/AI-Riksarkivet/htrflow
            cd htrflow

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
            git clone https://github.com/AI-Riksarkivet/htrflow
            cd htrflow

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


## Docker 

## docker-compose

## Helm

