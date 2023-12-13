include .env

VENV = venv
PYTHON = $(VENV)/Scripts/python
REPO_URL = https://Riksarkivet/htrflow
PACKAGE = htrflow
PIP = $(VENV)/Scripts/pip
ACTIVATE_ENV = . venv/bin/activate

.PHONY: help connect_to_repo release pre_commit venv local_install test_install local_dev_install tests-unit \
        mock-tests-unit tests-integration build local_build clean_venv local_clean_windows local_clean_linux \
        docker_build docs_install new_env_conda

## help - Display this help screen
help:
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e 's/\\$$//' -e 's/##//'

## connect_to_repo - Connects local repo to github repo with the same name as the repo locally
connect_to_repo:
	git init .
	git add --all
	git commit -m "First commit from CookieCutter"
	git branch -M main
	git remote add origin $(REPO_URL)
	git push -u origin main

## release - Release a new version
release:
	@echo git tag -a "vX.Y.Z" -m "Release X.Y.Z"
	@echo git push origin vX.Y.Z
	@echo change release vX.Y.Z in setup.py..

## pre_commit - Setup pre-commit hooks
pre_commit:
	$(PIP) install pre-commit
	pre-commit --version
	pre-commit install
	pre-commit run

## venv - Creates a venv, however, needs to be activated before use 
venv:
	python -m venv $(VENV)
	$(ACTIVATE_ENV)

## local_install - Installs python packages
local_install:
	$(PIP) install -e .

## test_install - Installs python packages for testing
test_install:
	pip install -e .
	pip install -r requirements_dev.txt

## local_dev_install - Install dev packages to test code
local_dev_install:
	$(PIP) install -r requirements_dev.txt

## tests-unit - Runs unit tests
tests-unit: local_dev_install
	pytest tests/unit -v

## mock-tests-unit - Mock unit tests
mock-tests-unit:
	pytest tests/unit -v

## tests-integration - Run integration tests
tests-integration:
	pytest test/integration

## build - Build python package
build:
	@echo "Building package"
	pip install build
	python -m build

## local_build - Build python package and remove cache files
local_build: build local_clean
	
## clean_venv - Removes Local venv and cache files
clean_venv: local_clean
	deactivate
	rm -rf $(VENV)

## local_clean_windows - Clean local folders for windows
local_clean_windows:
	if exist "dist" rmdir /S dist
	if exist ".pytest_cache" rmdir /S .pytest_cache
	if exist ".mypy_cache" rmdir /S .mypy_cache
	if exist ".tox" rmdir /S .tox
	if exist "src\${PACKAGE}.egg-info" rmdir /S src\${PACKAGE}.egg-info
	if exist "src\${PACKAGE}\__pycache__" rmdir /S src\${PACKAGE}\__pycache__
	if exist "tests\unit\__pycache__" rmdir /S tests\unit\__pycache__
	if exist ".coverage" del .coverage

## local_clean_linux - Clean local folders for linux
local_clean_linux:
	rm -r dist
	rm -r .pytest_cache
	rm -r .mypy_cache
	rm -r .tox
	rm -r src/${PACKAGE}.egg-info
	rm -r src/${PACKAGE}/__pycache__
	rm -r tests/unit/__pycache__
	rm .coverage

## docker_build - Build and run docker container
docker_build:
	docker build -t ${PACKAGE} .
	docker run ${PACKAGE}

## docs_install - Install requirements for mkdocs
docs_install:
	pip install 'mkdocs ==1.4.2'
	pip install 'mkdocs-material >=8.0.0'
	pip install 'mkdocs-jupyter ~=0.22.0'
	pip install 'mkdocstrings>=0.20.0'
	pip install 'mkdocstrings-python >= 0.8'

## new_env_conda - Create and setup new conda environment from environment.yml file
new_env_conda:
	@if conda env list | grep -q htrflow; then \
		conda env remove --name htrflow; \
	fi
	conda env create -f environment.yml

