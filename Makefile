.PHONY: ci check lint format test test-oldest install

ci: format lint check test

check:
	uv run pyright

lint:
	uvx ruff@0.15.22 check --fix

format:
	uvx ruff@0.15.22 format

test:
	uv run pytest --xdoctest

# Python 3.10 with every direct dependency at its minimum version, outside the lock.
test-oldest:
	rm -rf .venv-oldest && uv venv -q --python 3.10 .venv-oldest
	VIRTUAL_ENV=.venv-oldest uv pip install -q --resolution lowest-direct -e . --group dev
	.venv-oldest/bin/python -m pytest --xdoctest

install:
	uv sync --extra xarm --extra ability --extra realsense --extra retarget --extra vis --extra axis --group dev
