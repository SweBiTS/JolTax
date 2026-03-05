# Project Context: joltax

This file provides the necessary context for Gemini CLI to maintain the architectural integrity and development momentum of the `joltax` project.

## Project Overview
`joltax` is a vectorized Python library for querying and annotating the NCBI taxonomy (and derivatives like GTDB). It is designed to handle 200,000+ taxa annotations in seconds.

## Architectural Mandates
- **Vectorized Foundation:** All tree properties (parents, depths, ranks) MUST be stored in contiguous NumPy arrays.
- **Euler Tour Indexing:** Use entry/exit timestamps for $O(1)$ clade range queries.
- **Binary Lifting:** Use pre-calculated skip tables for $O(\log N)$ LCA and distance calculations.
- **Error Handling:** All TaxID-based query methods MUST implement a `strict=True` default. If a TaxID is missing from the tree, raise `TaxIDNotFoundError`. If `strict=False`, return standardized "safe" defaults (`None`, `[]`, or `-1`). Missing attributes for valid TaxIDs (e.g., common names) must return `None` and NOT raise an error.
- **Type Guards:** All public API methods MUST use explicit type guards (e.g., `isinstance(tax_id, (int, np.integer))`) to provide helpful `TypeError` messages for incorrect input types.
- **2025 Taxonomy Support:** Must handle the `superkingdom` vs `domain` shift. Ranks are auto-detected during build and enforced as mutually exclusive.
- **Persistence:** Use binary caches (NumPy `.npy` and Pickle for metadata) to avoid re-parsing `.dmp` files.
- **Modular Package Structure:** Maintain a clean separation between the tree logic (`joltree.py`), custom exceptions (`exceptions.py`), and taxonomic constants (`constants.py`).
- **Join-Safe API:** All methods returning DataFrames for mass annotation (e.g., `annotate`) MUST prefix taxonomic columns with `t_` and use `t_id` for TaxIDs to avoid name collisions during joins.
- **Taxonomy Integrity:** The `build_from_dmp` process should include explicit validation checks for tree integrity (e.g., detecting cycles, orphaned nodes, or multiple canonical ranks in a single lineage) to ensure downstream calculations (like NCA) remain reliable.

## Current State
- [x] Core `JolTree` implementation in `joltax/joltree.py`.
- [x] Vectorized `get_lineage`, `get_clade`, `get_clade_at_rank`, `get_lca`, and `get_distance`.
- [x] `annotate` for mass-annotation (benchmark: ~37s for 2.5M nodes).
- [x] Refactored `__init__` with `tax_dir` and standardized `strict=True` error handling.
- [x] Taxonomy Integrity: Explicit validation checks in `build_from_dmp`.
- [x] **VERSION 0.2.0 RELEASED:** Includes `t_` column prefixes, integrity checks, and modular code refactor.
- [x] Refactored package into modular files (`exceptions.py`, `constants.py`).
- [x] Provenance metadata (build time, source files, versioning).
- [x] Comprehensive test suite in `tests/` using **pytest**.
- [x] User documentation in `README.md` and `USAGE.md`.

## Pending Roadmap
(No pending core roadmap items at this time. Milestone 0.2.0 finalized all planned architectural features.)

## Technical Environment
- **Root Directory:** `/home/daniel/devel/JolTax`
- **Primary Module:** `joltax/joltree.py`
- **Dependencies:** `numpy`, `polars`, `rapidfuzz`
