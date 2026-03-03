<p align="center">
  <img src="https://raw.githubusercontent.com/SweBiTS/JolTax/main/assets/logo.png" alt="joltax logo" width="300">
</p>

# JolTax

**High-performance, vectorized taxonomy library for Python.**

JolTax is a Python library designed to handle massive taxonomies with extreme efficiency. By representing taxonomy trees as contiguous NumPy arrays and leveraging Polars for mass data handling, it achieves lightning-fast traversals, constant-time clade queries, and rapid mass annotation of large datasets.

## Key Features
- **Vectorized Performance:** Uses hardware-accelerated NumPy operations for million-scale property lookups.
- **Memory Efficient:** Optimized string store using Polars/Arrow reduces RAM footprint.
- **Fuzzy Name Search:** Rapid fuzzy matching using RapidFuzz to find TaxIDs from names.
- **Instant Clade Queries:** Quickly find all descendants of any node (even millions) using optimized range indexing.
- **Hyper-Vectorized LCA search:** Lowest Common Ancestor (LCA) search and node-to-node distance calculations at lightning speeds.
- **Mass Annotation:** Annotate massive TaxID tables with 2,000,000+ rows in under a second using Polars.

## Quick Start

```python
from joltax import JolTree

# Build and process the NCBI taxonomy
tree = JolTree(nodes='nodes.dmp', names='names.dmp')
# OR: tree = JolTree(tax_dir='/path/to/ncbi/taxonomy/')

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
pip install joltax
```

Requires: `numpy`, `polars`, `rapidfuzz`.

## Documentation

For a detailed API reference and a comprehensive "How-To" guide with example workflows, please see [USAGE.md](https://github.com/SweBiTS/JolTax/blob/main/USAGE.md).
