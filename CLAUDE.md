# CLAUDE.md

## CORE (NON-NEGOTIABLE)

1) Simplicity first
- Write the simplest correct code.
- Use straightforward control flow.
- Minimize lines of code. Fewer lines is a goal.

2) Validate every change
- After EVERY implementation, run the script.
- Report the exact run results EVERY time (command + output).
- No "should work" claims without a run log.

3) No safety nets
- Do NOT use try/except.
- Do NOT use hasattr.
- Do NOT add fallbacks or alternate paths.
- Assume inputs and environment are correct.
- Let the program crash on errors.

4) Function structure
- Prefer fewer, larger functions.
- Do NOT split into many small helper functions.
- Only create a new function when it removes real duplication and still keeps the total function count low.

## Notes

- Use `uv` to run scripts (not `python` or `pip` directly).
- GPU scripts: find a free GPU first and set `CUDA_VISIBLE_DEVICES`. Visualization scripts with infinite loops need a timeout.
- Type hints on function args and return types (except `None`, `Any`, `object`).
- Do NOT use `from __future__ import annotations`.
- Plan mode: do NOT read files speculatively or for general context. Only read a file if it is directly required to answer the task. Ask a clarifying question instead of reading files to guess intent.

## Commands

```shell
make install        # uv sync --all-extras
make ci             # format + lint + check + test
make format         # uvx ruff format
make lint           # uvx ruff check --fix
make check          # uv run pyright
make test           # uv run pytest && uv run xdoctest retargetingkit
```

## Testing

- Public functions should have doctests.
- Pytest: group test cases into a single class per feature.
- Use `torch.testing.assert_close` for numerical comparisons.
