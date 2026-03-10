"""Sphinx configuration for LITRIS API documentation."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath("../../..")
sys.path.insert(0, PROJECT_ROOT)

project = "LITRIS"
author = "cmb1705"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

autodoc_mock_imports = [
    "anthropic",
    "openai",
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
    "pymupdf",
    "pytesseract",
    "pdf2image",
    "pdf2image.exceptions",
    "PIL",
    "PIL.Image",
    "mcp",
    "keyring",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
