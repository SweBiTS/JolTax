# taxatree

A high-performance, vectorized NCBI taxonomy library designed for large-scale bioinformatics research.

## Features

- **Vectorized Performance:** Uses NumPy arrays for O(1) property lookups and rapid batch operations.
- **Mass Annotation:** Annotate hundreds of thousands of TaxIDs with names and canonical lineages in seconds.
- **Advanced Tree Operations:**
  - **Euler Tour Indexing:** Constant-time clade range queries.
  - **Binary Lifting:** Logarithmic-time Lowest Common Ancestor (LCA) and distance calculations.
- **Efficient Memory Management:** Can save/load pre-processed binary caches to eliminate parsing delays (DMP files).
- **Service-Ready:** Optimized for high-concurrency research servers.

## Installation

```bash
pip install .
```

Requires:
- Python >= 3.8
- numpy
- pandas

## Quick Start

```python
from taxatree import TaxonomyTree

# Build from NCBI DMP files
tree = TaxonomyTree(nodes_file='nodes.dmp', names_file='names.dmp')

# Or load from a pre-built binary cache
# tree = TaxonomyTree.load('taxonomy_cache')

# Get full lineage for a TaxID
lineage = tree.get_lineage(562)  # [1, 2, 1224, ..., 562]

# Mass annotation of a list of TaxIDs
import pandas as pd
tax_ids = [562, 561, 1236]
df = tree.annotate_table(tax_ids)
print(df)

# Fast clade query
bacteria_clade = tree.get_clade(2)  # Returns a NumPy array of all TaxIDs in Bacteria

# Save binary cache for faster loading next time
tree.save('taxonomy_cache')
```

## Why use taxatree?

Traditional object-oriented taxonomy trees in Python (storing each node as a Python object) can consume several gigabytes of RAM and take minutes to load the full NCBI taxonomy. `taxatree` uses contiguous memory, reducing RAM overhead by up to 90% and making tree traversals hardware-accelerated.
