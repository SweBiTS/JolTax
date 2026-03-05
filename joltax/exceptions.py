"""
joltax/exceptions.py
Custom exceptions for the joltax library.
"""

class TaxIDNotFoundError(Exception):
    """Raised when a requested NCBI TaxID is not found in the taxonomy tree."""
    pass

class TaxonomyIntegrityError(Exception):
    """Raised when the taxonomy tree structure is inconsistent or corrupt (e.g., cycles, orphans)."""
    pass
