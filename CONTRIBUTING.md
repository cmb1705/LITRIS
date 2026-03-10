# Contributing

Thank you for contributing to LITRIS. This guide covers development setup,
code style, testing, and the pull request workflow.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Issue Tracking (bd)

This project uses beads (bd) for issue tracking. Start by finding ready work,
then mark issues in progress before you begin.

```bash
bd ready
bd update <id> --status in_progress
```

When finished:

```bash
bd close <id>
```

## Pre-commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

The hooks run:

- Trailing whitespace and EOF fixes
- YAML validation
- Large file detection
- Ruff linting and formatting
- mypy type checking (on src/)
- pytest quick tests (on push only)

To run all hooks manually:

```bash
pre-commit run --all-files
```

## Code Style

- Follow PEP 8 and use type hints.
- Add docstrings to public modules, classes, and functions.
- Avoid emojis in code and documentation.
- Run ruff for linting and formatting:

```bash
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/
```

## Testing

Run unit tests locally before opening a PR:

```bash
pytest --tb=short -q
```

Integration tests require a local Zotero database and are skipped by default.
To run them explicitly:

```bash
pytest -m integration --tb=short -q
```

## Documentation

- Use Markdown with a single H1 per file.
- Add blank lines around headings, lists, and code fences.
- Use fenced code blocks with language tags.

To build API documentation with Sphinx:

```bash
python -m pip install -r docs/requirements.txt
cd docs/sphinx
make html
```

## Continuous Integration

Pull requests trigger GitHub Actions CI which runs:

| Job | Description |
|-----|-------------|
| test | Runs pytest on Python 3.10-3.12 across Ubuntu, Windows, and macOS |
| type-check | Runs mypy type checking on src/ |
| security | Runs pip-audit to check for known vulnerabilities |

All jobs must pass before merging. Check the Actions tab for details on failures.

## Pull Requests

- Create a focused branch per issue.
- Keep PRs small and scoped to one issue.
- Include a clear description and testing notes.
- Update or add documentation for behavior changes.
- Do not add attribution lines to commits (for example, "Generated with ...").
