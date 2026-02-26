# Project Context: taxatree

This file provides the necessary context for Gemini CLI to maintain the architectural integrity and development momentum of the `taxatree` project.

## Project Overview
`taxatree` is a high-performance, vectorized Python library for querying and annotating the NCBI taxonomy (and derivatives like GTDB). It was born from a need to handle 200,000+ taxa annotations in seconds, a task that traditional object-oriented trees handle poorly.

## Architectural Mandates
- **Vectorized Foundation:** All tree properties (parents, depths, ranks) MUST be stored in contiguous NumPy arrays. Avoid per-node Python objects.
- **Euler Tour Indexing:** Use entry/exit timestamps for $O(1)$ clade range queries.
- **Binary Lifting:** Use pre-calculated skip tables for $O(\log N)$ LCA and distance calculations.
- **2025 Taxonomy Support:** Must handle the `superkingdom` vs `domain` shift. Ranks are auto-detected during build and enforced as mutually exclusive.
- **Persistence:** Use binary caches (NumPy `.npy` and Pickle for metadata) to avoid re-parsing `.dmp` files.

## Current State
- [x] Core `TaxonomyTree` implementation in `taxatree/tree.py`.
- [x] Vectorized `get_lineage`, `get_clade`, `get_clade_at_rank`, `get_lca`, and `get_distance`.
- [x] `annotate_table` for mass-annotation (benchmark: ~37s for 2.5M nodes).
- [x] Provenance metadata (build time, source files, versioning).
- [x] Basic test suite in `tests/test_tree.py`.
- [x] User documentation in `README.md` and `USAGE.md`.

## Pending Roadmap
1. **Refactor StringMeUp:** Update the original `StringMeUp` repository to use `taxatree` as a dependency and remove its internal `taxonomy.py`.
2. **FastAPI Service (`taxa-api`):** Create a service layer to host the taxonomy in memory on a research server, providing a REST API for remote queries.
3. **Advanced Vectorization:** Potentially pre-calculate canonical rank maps to further accelerate `annotate_table` (reducing it from 37s to <2s for the full taxonomy).

## Technical Environment
- **Root Directory:** `/home/daniel/devel/taxatree`
- **Primary Module:** `taxatree/tree.py`
- **Dependencies:** `numpy`, `pandas`
