# taxatree

**High-performance, vectorized NCBI taxonomy library for large-scale bioinformatics research.**

`taxatree` is a Python library designed to handle the massive NCBI taxonomy (and derivatives like GTDB) with extreme efficiency. By replacing traditional object-oriented trees with contiguous NumPy arrays, it achieves lightning-fast traversals, constant-time clade queries, and rapid mass annotation of large datasets.

## Key Features

- **Vectorized Performance:** Uses hardware-accelerated NumPy operations for O(1) property lookups.
- **2025 Taxonomy Ready:** Native support for the recent NCBI/GTDB shift from `superkingdom` to `domain`.
- **Euler Tour Indexing:** Instant clade range queries (even for millions of nodes).
- **Binary Lifting (Skip Tables):** Logarithmic-time Lowest Common Ancestor (LCA) and distance calculations.
- **Mass Annotation:** Annotate tables with 200,000+ rows in under a second.
- **Full Provenance:** Binary caches store build timestamps, source file paths, and package versions for reproducible research.

## Quick Start

```python
from taxatree import TaxonomyTree

# Build and process the NCBI taxonomy
tree = TaxonomyTree(nodes_file='nodes.dmp', names_file='names.dmp')

# Save for instant loading next time
tree.save('my_taxonomy_cache')

# Re-load in seconds
tree = TaxonomyTree.load('my_taxonomy_cache')

# Get a lineage
lineage = tree.get_lineage(9606)  # [1, ..., 9606]

# Get all genera in the Bacteria clade
bacteria_genera = tree.get_clade_at_rank(2, 'genus')
```

## Installation

```bash
cd taxatree
pip install .
```

Requires: `numpy`, `pandas`.

## Documentation

For a detailed API reference and a comprehensive "How-To" guide with example workflows, please see [USAGE.md](./USAGE.md).
