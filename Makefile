.PHONY: requirements
requirements:
	pip install -r requirements/base.txt

.PHONY: requirements_tools
requirements_tools:
	pip install -r requirements/tools.txt

.PHONY: requirements_test
requirements_test:
	pip install -r requirements/test.txt

.PHONY: requirements_notebooks
requirements_notebooks:
	pip install -r requirements/notebooks.txt

.PHONY: typecheck
typecheck: requirements_tools
	mypy bandito tests

.PHONY: lint
lint: requirements_tools
	flake8 --exclude=env,venv
	black --line-length 120 --check .

.PHONY: autoformat
autoformat: requirements_tools
	black --line-length 120 .

.PHONY: run_jupyter
run_jupyter: requirements_notebooks
	jupyter notebook
