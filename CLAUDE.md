# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`dspy-toolbox` is a Python library providing reasoning and debugging utilities for DSPy prompts.

## Setup

This is a Python project. Common package managers found in the `.gitignore` include `uv`, `poetry`, `pdm`, `pipenv`, and `pixi`. Once a package manager and dependencies are added, update this file with the actual install/run commands.

## Development Commands

- Install deps: `pip install -r requirements.txt -r requirements-dev.txt`
- Run tests: `pytest` / `pytest tests/path/to/test_file.py::TestClass::test_method` for a single test
- Lint: `python -m flake8 src/`
- Type check: `python -m mypy src/`

## Code Quality Requirements

All three checks **must pass** before any commit. Do not commit code that fails any of them.

```bash
python -m flake8 src/
python -m mypy src/
pytest
```

- `flake8` — configured in `setup.cfg` under `[flake8]`; max line length 120, wemake-python-styleguide rules apply
- `mypy` — configured in `setup.cfg` under `[mypy]`; strict mode (`disallow_untyped_defs`, `disallow_any_unimported`, `disallow_subclassing_any`, etc.)
- `pytest` — test coverage must stay at **95% or above**; enforced automatically via `--cov-fail-under=95` in `pytest.ini`

## Coding Conventions

1. **No raw dicts for structured data** — every JSON-shaped structure (function parameters, return values, model fields, API payloads, etc.) must be represented by a Pydantic `BaseModel`. Never use `dict` as a public interface boundary.

2. **No magic strings** — every string constant must be either a `StrEnum` member (for a closed set of values, e.g. field types, statuses, categories) or a named module-level constant (e.g. `_DEFAULT_THRESHOLD: float = 0.85`). Bare string literals in logic are not allowed.

3. **Strict type annotations** — every function and method must have a complete type signature: all parameters annotated (including `self` is exempt, but all others must be typed) and a return type declared. No `def foo(x)` or `def foo() -> None` with untyped parameters. This is enforced by mypy (`disallow_untyped_defs = True`, `disallow_incomplete_defs = True`).

4. **f-strings for all string formatting** — use f-strings (`f"..."`) exclusively. Never use `%`-style (`"hello %s" % name`) or `str.format()` (`"hello {}".format(name)`). This applies to log messages, error strings, and all other string construction.

5. **Instrument all slow operations with `timed`** — any call that is time- or resource-intensive must be wrapped with the `timed` context manager from `proc.base.timing`. This includes, but is not limited to:
   - LLM / model inference calls (e.g. `model(...)`, `model.generate(...)`, HuggingFace `forward` passes)
   - Captum attribution passes (`lig.attribute(...)`)
   - Model loading from disk (`torch.load`, `from_pretrained`, `load_state_dict`)
   - Training evaluation steps (`estimate_loss` or equivalent)
   - Any network I/O or external API call expected to take more than a few hundred milliseconds

   Usage:
   ```python
   from proc.base.timing import timed

   with timed("descriptive_label", logger=self._logger):
       result = slow_operation(...)
   ```

   Always pass `logger=` explicitly so timing messages are routed through the caller's logger, not the library default. Use a label that uniquely identifies the call site (include loop indices or parameter values where helpful, e.g. `f"example[{i}] _run_lig"`).