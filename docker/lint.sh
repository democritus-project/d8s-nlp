#!/usr/bin/env bash

set -euxo pipefail

echo "Running linters and formatters..."

isort d8s_nlp/ tests/

black d8s_nlp/ tests/

mypy d8s_nlp/ tests/

pylint d8s_nlp/*.py

flake8 d8s_nlp/ tests/

bandit -r d8s_nlp/

# we run black again at the end to undo any odd changes made by any of the linters above
black d8s_nlp/ tests/
