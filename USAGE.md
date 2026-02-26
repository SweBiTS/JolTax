# taxatree: User Guide & API Documentation

This guide provides a deep dive into the `taxatree` package, its core algorithms, and typical research workflows.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Workflow: How-To Guide](#workflow-how-to-guide)
3. [API Reference](#api-reference)
4. [The 2025 Taxonomy Standard](#the-2025-taxonomy-standard)

---

## Core Concepts

### Vectorization
Unlike traditional libraries that store each TaxID as a separate Python object (consuming gigabytes of RAM), `taxatree` stores the entire taxonomy in contiguous NumPy arrays. A TaxID becomes a direct index into these arrays, allowing for O(1) attribute lookups and hardware-accelerated batch operations.

### Euler Tour Indexing
To make clade queries (getting all descendants of a node) instantaneous, `taxatree` performs a one-time traversal to assign **Entry** and **Exit** timestamps to every node. A node $v$ is a descendant of $u$ if and only if $entry[u] \le entry[v] \le exit[u]$. This turns a complex tree traversal into a simple numeric range query.

### Binary Lifting (Skip Tables)
Finding the Lowest Common Ancestor (LCA) is optimized using binary lifting. Instead of walking up one step at a time, each node stores its $2^k$-th ancestor (2nd, 4th, 8th, 16th, etc.). This allows finding the LCA of any two nodes in $O(\log N)$ time.

---

## Workflow: How-To Guide

A typical research workflow involving `taxatree` consists of three stages: **Build**, **Query**, and **Annotate**.

### 1. The Build-and-Cache Phase
You only need to build the tree once from NCBI or GTDB `.dmp` files.

```python
import logging
from taxatree import TaxonomyTree

# Enable logging to see the build progress
logging.getLogger('taxatree').setLevel(logging.INFO)

# Initial build (takes ~90 seconds for full NCBI)
tree = TaxonomyTree(
    nodes_file='taxonomy/nodes.dmp', 
    names_file='taxonomy/names.dmp'
)

# Save to a binary cache directory
tree.save("ncbi_cache")
```

### 2. The Daily Research Phase (Fast Loading)
In your daily scripts, you can skip the slow DMP parsing and load the processed cache in under a second.

```python
from taxatree import TaxonomyTree

# Near-instant load
tree = TaxonomyTree.load("ncbi_cache")
```

### 3. Analyzing a Clade
Suppose you are studying the diversity of a specific genus or family.

```python
# 1. Get all nodes within the Bacteria clade (GTDB ID: 5016879)
bacteria_nodes = tree.get_clade(5016879)
print(f"Total nodes in Bacteria: {len(bacteria_nodes):,}")

# 2. Get only the species within that clade
bacteria_species = tree.get_clade_at_rank(5016879, 'species')
print(f"Total unique species in Bacteria: {len(bacteria_species):,}")
```
**Expected Output:**
```text
Total nodes in Bacteria: 352,410
Total unique species in Bacteria: 84,212
```

### 4. Annotating a Table
This is the most common use case for research: turning a column of TaxIDs into a full taxonomic table.

```python
# List of 200,000 TaxIDs from a classification file
import numpy as np
tax_ids = np.random.choice(tree._index_to_id, 200000)

# Mass annotation
df = tree.annotate_table(tax_ids)

# Save to CSV for downstream analysis
df.to_csv("annotated_results.csv", index=False)
```
**Expected Output:**
The resulting DataFrame (`df`) will have columns ordered for maximum readability:
`tax_id` | `domain` | `kingdom` | ... | `scientific_name` | `rank`

---

## API Reference

### Initialization & Persistence

#### `TaxonomyTree(nodes_file=None, names_file=None)`
- **`nodes_file`**: Path to `nodes.dmp`.
- **`names_file`**: Path to `names.dmp`.
- *Note:* If files are provided, the tree is built immediately. If not, you must use `load()`.

#### `@classmethod load(directory)`
Loads a pre-processed binary cache from the specified directory. This is the recommended way to use `taxatree` in production.

#### `save(directory)`
Saves the internal arrays and provenance metadata to a directory.

---

### Taxonomic Queries

#### `get_lineage(tax_id: int) -> List[int]`
Returns the full path from the root (ID: 1) to the given TaxID.
- *Performance:* Fast O(depth) lookup.

#### `get_clade(tax_id: int) -> List[int]`
Returns a list of all TaxIDs (descendants) rooted at the given node.
- *Performance:* Instant O(clade_size) range query.

#### `get_clade_at_rank(tax_id: int, rank_name: str) -> List[int]`
Returns all descendants of `tax_id` that belong to the specified rank.
- *Example:* `get_clade_at_rank(2, 'genus')`.
- *Performance:* Vectorized mask operation.

#### `get_lca(tax_id_1: int, tax_id_2: int) -> int`
Finds the Lowest Common Ancestor of two nodes.
- *Performance:* O(log depth) via Binary Lifting.

#### `get_distance(tax_id_1: int, tax_id_2: int) -> int`
Returns the number of edges (hops) between two nodes in the tree.
- *Performance:* O(log depth).

---

### Mass Operations

#### `annotate_table(tax_ids: List[int]) -> pandas.DataFrame`
Produces a formatted DataFrame with the following columns:
- `tax_id`: The input ID.
- `domain` / `superkingdom`: Consolidated top-level rank.
- `kingdom` through `species`: The canonical ranks.
- `scientific_name`: The scientific name of the taxon itself.
- `rank`: The original rank string.

---

## The 2025 Taxonomy Standard
As of early 2025, the NCBI Taxonomy has shifted away from `superkingdom` in favor of `domain` for cellular life. 

`taxatree` handles this by:
1. **Auto-Detection:** During the build, it identifies which rank is used as the top level.
2. **Mutual Exclusivity:** It enforces that a taxonomy uses either `superkingdom` OR `domain` as its top-level identifier, but not both.
3. **Dynamic Columns:** The `annotate_table` output dynamically renames its first taxonomic column based on what it detected in your specific database.
