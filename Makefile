all: lint mypy test

lint:
	pylint vectara_eval || true
	flake8 vectara_eval tests || true
mypy:
	mypy vectara_eval || true

test:
	python -m unittest discover -s tests -b

.PHONY: all lint mypy test