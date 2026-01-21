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

## Pull Requests

- Create a focused branch per issue.
- Keep PRs small and scoped to one issue.
- Include a clear description and testing notes.
- Update or add documentation for behavior changes.
- Do not add attribution lines to commits (for example, "Generated with ...").
