all: lint mypy test

lint:
	pylint open_rag_eval || true
	flake8 open_rag_eval tests || true
mypy:
	mypy open_rag_eval || true

test:
	TRANSFORMERS_VERBOSITY=error python -m unittest discover -s tests -b

.PHONY: all lint mypy test