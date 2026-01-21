"""Reference manager interfaces and implementations."""

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.references.factory import create_reference_db, get_available_providers

__all__ = [
    "BaseReferenceDB",
    "ReferenceProvider",
    "create_reference_db",
    "get_available_providers",
]
