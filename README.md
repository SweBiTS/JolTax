# joltax

**High-performance, vectorized NCBI taxonomy library for large-scale bioinformatics research.**

`joltax` is a Python library designed to handle the massive NCBI taxonomy (and derivatives like GTDB) with extreme efficiency. By replacing traditional object-oriented trees with contiguous NumPy arrays and leveraging Polars for mass data handling, it achieves lightning-fast traversals, constant-time clade queries, and rapid mass annotation of large datasets.

## Key Features
- **Vectorized Performance:** Uses hardware-accelerated NumPy operations for million-scale property lookups.
- **Memory Efficient:** Optimized string store using Polars/Arrow reduces RAM footprint by ~70% compared to standard Python dicts.
- **2025 Taxonomy Ready:** Native support for the recent NCBI/GTDB shift from `superkingdom` to `domain`.
- **Fuzzy Name Search:** Rapid, rank-aware fuzzy matching using RapidFuzz to find TaxIDs from names.
- **Euler Tour Indexing:** Instant clade range queries (even for millions of nodes).
- **Hyper-Vectorized LCA:** Batch Lowest Common Ancestor (LCA) and distance calculations at million-query-per-second speeds.
- **Mass Annotation:** Annotate tables with 2,000,000+ rows in under a second using Polars.
- **Full Provenance:** Binary caches store build timestamps, version validation, and source file paths.

## Quick Start

```python
from joltax.joltree import JolTree

# Build and process the NCBI taxonomy
tree = JolTree(nodes_file='nodes.dmp', names_file='names.dmp')

# Save for instant loading next time
tree.save('my_taxonomy_cache')

# Re-load in milliseconds (using zero-copy Arrow IPC)
tree = JolTree.load('my_taxonomy_cache')

# Batch LCA (process 10,000 pairs in <10ms)
lcas = tree.get_lca_batch(ids1, ids2)

# Fuzzy search for a name (returns a Polars DataFrame)
results = tree.search_name('Escherchia', fuzzy=True)
print(results)
```

## Installation

```bash
cd joltax
pip install .
```

Requires: `numpy`, `polars`, `psutil`, `rapidfuzz`.
pip install .
```

Requires: `numpy`, `polars`, `pandas`, `rapidfuzz`.

## Documentation

For a detailed API reference and a comprehensive "How-To" guide with example workflows, please see [USAGE.md](./USAGE.md).
