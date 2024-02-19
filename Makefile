REPO_URL = https://github.com/Swedish-National-Archives-AI-lab/htrflow_core
PACKAGE = htrflow_core

## help - Display this help screen in a more structured format
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


connect_to_repo: ## connect_to_repo - Connects local repo to github repo with the same name as the repo locally
	git init .
	git add --all
	git commit -m "First commit.."
	git branch -M main
	git remote add origin $(REPO_URL)
	git push -u origin main

configure_startup: ## configure_startup - Configuring poetry for venv + Installing project dependencies with poetry + Activate venv
	pip install --quiet --upgrade pip poetry
	poetry config --local virtualenvs.in-project true
	poetry install
	poetry shell


magic: connect_to_repo configure_startup ## runs connect_to_repo & configure_startup

pre_commit: ## pre_commit - Setup pre-commit hooks
	pre-commit --version
	pre-commit install
	pre-commit run

local_clean_linux: ## local_clean_linux - Clean local folders for linux
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -f .coverage
	rm -rf src/${PACKAGE}/__pycache__
	rm -rf tests/unit/__pycache__
	rm -rf dist
	rm -rf src/${PACKAGE}.egg-info

docker_build: ## docker_build - Build and run docker container
	docker build -t ${PACKAGE} .
	docker run ${PACKAGE}

patch: ## patch - how to bump version
	@echo poetry version prerelease
	@echo poetry version patch

quality: ## quality - Check code quality
	ruff check .
	mypy . 