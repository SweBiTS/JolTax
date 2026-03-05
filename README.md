<p align="center">
  <img src="https://raw.githubusercontent.com/SweBiTS/JolTax/main/assets/logo.png" alt="joltax logo" width="300">
</p>

<p align="center">
  <a href="https://anaconda.org/bioconda/joltax"><img src="https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square" alt="Bioconda"></a>
  <a href="https://pypi.org/project/joltax/"><img src="https://img.shields.io/pypi/v/joltax.svg?style=flat-square" alt="PyPI version"></a>
</p>

# JolTax

**A vectorized taxonomy library for Python.**

JolTax is a tool for working with large taxonomies like NCBI or GTDB. It stores the entire tree in contiguous NumPy arrays, enabling fast (like a *jolt*) traversals, clade queries, and mass annotation of datasets using Polars.

## Key Features
- **Search:** Exact and approximate matching to find TaxIDs from strings.
- **Clade Queries:** Instantly identify all descendants of a TaxID.
- **Annotate:** Instantly return a Polars dataframe with the complete canonical taxonomy (for any number of TaxIDs), for easy annotation of your own datasets. 
- **Batch Processing:** Get Lowest Common Ancestor (LCA) and node-to-node distances for thousands of TaxID pairs at once.
- **Array-Based Core:** Uses NumPy operations for property lookups and tree traversals.
- **Pre-build:** Build and save (cache) your taxonomies for instant loading later.

If you prefer an interactive experience for building and exploring taxonomies, a command-line interface is also available:
[**JolTax-CLI**](https://github.com/SweBiTS/JolTax-CLI).

## Installation

### From Bioconda (Recommended)
```bash
conda install -c bioconda joltax
```

### From PyPI
```bash
pip install joltax
```

Requires: `numpy`, `polars`, `rapidfuzz`.

## Quick Start

```python
from joltax import JolTree

# Build from NCBI DMP files (dir where names.dmp and nodes.dmp are)
tree = JolTree('/path/to/ncbi/taxonomy/')

# Save a binary cache in dir "taxonomy_cache" for instant loading later
tree.save('taxonomy_cache')

# Load the cache
tree = JolTree.load('taxonomy_cache')

# Find a TaxID by name (fuzzy=False by default)
results = tree.search_name('Escherchia', fuzzy=True)

# Annotate a list of TaxIDs with their full canonical rank lineages
# Returns a Polars DataFrame with columns prefixed by 't_' (e.g., t_phylum, t_genus)
df = tree.annotate([9606, 562])

# Batch LCA calculation
lcas = tree.get_lca_batch(ids1, ids2)
```

## Documentation

For a detailed API reference and a step-by-step guide, see [USAGE.md](https://github.com/SweBiTS/JolTax/blob/main/USAGE.md).
