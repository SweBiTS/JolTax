# joltax: User Guide & API Documentation

This guide provides a deep dive into the `joltax` package, its core algorithms, and typical research workflows.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Workflow: How-To Guide](#workflow-how-to-guide)
3. [API Reference](#api-reference)
4. [The 2025 Taxonomy Standard](#the-2025-taxonomy-standard)

---

## Core Concepts

### Vectorization
Unlike traditional libraries that store each TaxID as a separate Python object (consuming gigabytes of RAM), `joltax` stores the entire taxonomy in contiguous NumPy arrays. A TaxID becomes a direct index into these arrays, allowing for O(1) attribute lookups and hardware-accelerated batch operations.

### Memory-Optimized Metadata (v0.1.1)
Starting in version 0.1.1, all metadata (names, ranks) is stored in a **vectorized String Store** using Polars and Apache Arrow IPC. This reduces the RAM footprint by ~70% and enables zero-copy loading from disk.

### Euler Tour Indexing
To make clade queries (getting all descendants of a node) instantaneous, `joltax` performs a one-time traversal to assign **Entry** and **Exit** timestamps to every node. A node $v$ is a descendant of $u$ if and only if $entry[u] \le entry[v] \le exit[u]$. This turns a complex tree traversal into a simple numeric range query.

### Hyper-Vectorized Binary Lifting (Skip Tables)
Finding the Lowest Common Ancestor (LCA) is optimized using binary lifting. Instead of walking up one step at a time, each node stores its $2^k$-th ancestor (2nd, 4th, 8th, 16th, etc.). In version 0.1.1, the table is transposed and hyper-vectorized to support **million-query-per-second batch processing**.

---

## Workflow: How-To Guide

A typical research workflow involving `joltax` consists of three stages: **Build**, **Query**, and **Annotate**.

### 1. The Build-and-Cache Phase
You only need to build the tree once from NCBI or GTDB `.dmp` files.

```python
import logging
from joltax.joltree import JolTree

# Enable logging to see the build progress
logging.getLogger('joltax').setLevel(logging.INFO)

# Initial build (takes ~60-90 seconds for full NCBI)
tree = JolTree(
    nodes_file='taxonomy/nodes.dmp', 
    names_file='taxonomy/names.dmp'
)

# Save to a binary cache directory (uses Arrow IPC for strings)
tree.save("ncbi_cache")
```

### 2. The Daily Research Phase (Fast Loading)
In your daily scripts, you can load the processed cache in under a second. `joltax` automatically validates the cache version to ensure compatibility.

```python
from joltax.joltree import JolTree

# Near-instant load with zero-copy Arrow IPC
tree = JolTree.load("ncbi_cache")

# PERFORMANCE TIP: For high-throughput API use, pre-warm the LCA cache
tree._ensure_up_table()
```

### 3. Searching for Taxa by Name
You can find TaxIDs using exact or fuzzy matching via the optimized Polars search index.

```python
# 1. Exact match (O(log N) lookup)
df = tree.search_name("Escherichia coli")

# 2. Fuzzy match (handles typos, returns ranked candidates)
# Results are boosted for canonical ranks (Genus, Species, etc.)
results = tree.search_name("Escherchia", fuzzy=True, limit=5)
print(results)
```

### 4. Analyzing a Clade
Suppose you are studying the diversity of a specific genus or family.

```python
# 1. Get all nodes within the Bacteria clade (GTDB ID: 5016879)
bacteria_nodes = tree.get_clade(5016879)
print(f"Total nodes in Bacteria: {len(bacteria_nodes):,}")

# 2. Get only the species within that clade
bacteria_species = tree.get_clade_at_rank(5016879, 'species')
print(f"Total unique species in Bacteria: {len(bacteria_species):,}")
```

### 5. High-Throughput LCA & Distance (Batching)
For large-scale comparisons, avoid Python loops and use the vectorized batch methods.

```python
import numpy as np

# Resolve 100,000 LCA pairs in ~10-20ms
ids1 = np.random.choice(tree._index_to_id, 100000)
ids2 = np.random.choice(tree._index_to_id, 100000)

lcas = tree.get_lca_batch(ids1, ids2)
dists = tree.get_distance_batch(ids1, ids2)
```

### 6. Annotating a Table
This is the most common use case for research: turning a column of TaxIDs into a full taxonomic table. `joltax` uses **Polars** for lightning-fast mass annotation.

```python
# List of 1,000,000 TaxIDs
import numpy as np
tax_ids = np.random.choice(tree._index_to_id, 1000000)

# Mass annotation (under 1 second)
df = tree.annotate_table(tax_ids)

# Save to Parquet or CSV using Polars
df.write_parquet("annotated_results.parquet")
```

---

## API Reference

### Initialization & Persistence

#### `JolTree(nodes_file=None, names_file=None)`
- **`nodes_file`**: Path to `nodes.dmp`.
- **`names_file`**: Path to `names.dmp`.
- *Note:* If files are provided, the tree is built immediately. If not, you must use `load()`.

#### `@classmethod load(directory)`
Loads a pre-processed binary cache. Raises `RuntimeError` if the cache was built with an incompatible version of `joltax`.

#### `save(directory)`
Saves the internal arrays, pre-calculated canonical maps, and name indices. Uses Arrow IPC for string metadata.

---

### Taxonomic Queries

#### `get_name(tax_id: int) -> str`
Returns the scientific name of the given TaxID.

#### `get_common_name(tax_id: int) -> Optional[str]`
Returns the GenBank common name, if available.

#### `search_name(query: str, fuzzy: bool = False, limit: int = 10) -> polars.DataFrame`
Finds TaxIDs matching the query string using the optimized Polars search index.

#### `get_lineage(tax_id: int) -> List[int]`
Returns the full path from the root (ID: 1) to the given TaxID.

#### `get_clade(tax_id: int) -> List[int]`
Returns a list of all TaxIDs (descendants) rooted at the given node.

#### `get_clade_at_rank(tax_id: int, rank_name: str) -> List[int]`
Returns all descendants of `tax_id` that belong to the specified rank.

#### `get_lca(tax_id_1: int, tax_id_2: int) -> int`
Finds the Lowest Common Ancestor of two nodes in $O(\log N)$ time.

#### `get_lca_batch(ids1: np.ndarray, ids2: np.ndarray) -> np.ndarray`
Hyper-vectorized batch LCA calculation. Processes millions of pairs per second.

#### `get_distance(tax_id_1: int, tax_id_2: int) -> int`
Returns the number of edges (hops) between two nodes.

#### `get_distance_batch(ids1: np.ndarray, ids2: np.ndarray) -> np.ndarray`
Vectorized batch distance calculation.

---

### Mass Operations

#### `annotate_table(tax_ids: List[int]) -> polars.DataFrame`
Produces a formatted Polars DataFrame using vectorized lookups and Polars `gather`.

---

## The 2025 Taxonomy Standard
As of early 2025, the NCBI Taxonomy has shifted away from `superkingdom` in favor of `domain` for cellular life. 

`joltax` handles this by:
1. **Auto-Detection:** During the build, it identifies which rank is used as the top level.
2. **Mutual Exclusivity:** It enforces that a taxonomy uses either `superkingdom` OR `domain` as its top-level identifier, but not both.
3. **Dynamic Columns:** The `annotate_table` output dynamically renames its first taxonomic column based on what it detected in your specific database.
