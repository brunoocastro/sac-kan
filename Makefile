lint:
	black . --config=pyproject.toml
	isort . --profile black
	flake8 . --config=setup.cfg

setup:
	scripts/setup.sh