#!/usr/bin/env python3
"""
joltax/joltree.py
Implementation of a high-performance, vectorized taxonomy tree.
"""

__version__ = "0.2.0"

# The minimum version of a saved taxonomy cache that is compatible with this software.
# Increment this when making breaking changes to the binary layout or metadata structure.
MINIMUM_CACHE_VERSION = "0.1.1"

import logging
import os
import datetime
from typing import Dict, List, Optional, Set, Union, Tuple, Any
from collections import namedtuple

import numpy as np
import polars as pl
from rapidfuzz import process, fuzz, utils

from .constants import CANONICAL_RANKS, RANK_TO_CODE
from .exceptions import TaxIDNotFoundError, TaxonomyIntegrityError

# Set up logging for the module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class JolTree:
    """
    A high-performance taxonomy representation using vectorized arrays.
    
    This class uses contiguous NumPy arrays for fast lookups, traversals, and mass annotations.
    It leverages Polars for memory-efficient string storage and batch processing.
    
    Attributes:
        parents (np.ndarray): Array where parents[i] is the internal index of the parent of node i.
        depths (np.ndarray): Array where depths[i] is the distance from root to node i.
        ranks (np.ndarray): Array where ranks[i] is the numeric index of the rank of node i.
        rank_names (List[str]): List of rank names corresponding to indices in `ranks`.
        top_rank (str): Detected top-level rank ('superkingdom' or 'domain').
        canonical_maps (Dict[str, np.ndarray]): Maps canonical rank names to arrays of ancestor indices.
    """

    def __init__(self, tax_dir: Optional[str] = None, nodes: Optional[str] = None, names: Optional[str] = None):
        """
        Initialize the taxonomy tree. 
        
        If files or a directory are provided, it builds the vectorized structure 
        from NCBI DMP files. Otherwise, it initializes an empty tree.
        
        Args:
            tax_dir: Path to a directory containing both nodes.dmp and names.dmp.
            nodes: Path to NCBI nodes.dmp.
            names: Path to NCBI names.dmp.
        """
        # Vectorized internal index mapping (sorted array of TaxIDs)
        self._index_to_id: np.ndarray = np.array([], dtype=np.int32)
        
        # Primary arrays (indexed by the dense internal index)
        self.parents: np.ndarray = np.array([], dtype=np.int32)
        self.depths: np.ndarray = np.array([], dtype=np.int32)
        self.ranks: np.ndarray = np.array([], dtype=np.uint8)
        
        # Metadata storage (Polars Series for memory efficiency)
        self._scientific_names: pl.Series = pl.Series("scientific_name", [], dtype=pl.String)
        self._common_names: pl.Series = pl.Series("common_name", [], dtype=pl.String)
        self.rank_names: List[str] = []
        self.top_rank: str = "domain"  # Default, will be detected
        self._source_nodes: Optional[str] = None
        self._source_names: Optional[str] = None
        self._build_time: Optional[str] = None
        
        # Clade query support (Euler Tour timestamps)
        self.entry_times: np.ndarray = np.array([], dtype=np.int32)
        self.exit_times: np.ndarray = np.array([], dtype=np.int32)
        
        # Binary lifting table for LCA (initialized on demand)
        self._up_table: Optional[np.ndarray] = None
        
        # Pre-calculated canonical rank maps (dense internal index -> dense internal index)
        # Values are internal indices, not TaxIDs. -1 means no ancestor at that rank.
        self.canonical_maps: Dict[str, np.ndarray] = {}
        
        # Search index (Polars DataFrame: name -> tax_id)
        self._search_index: pl.DataFrame = pl.DataFrame(schema={"name": pl.String, "tax_id": pl.Int32})
        
        # Caches for vectorized lookup (prepared during build/load)
        self._sci_names_lookup: Optional[pl.Series] = None
        self._rank_names_series: Optional[pl.Series] = None
        self._ranks_extended: Optional[np.ndarray] = None
        
        # Resolve paths
        nodes_path = nodes
        names_path = names
        
        if tax_dir:
            if not os.path.isdir(tax_dir):
                raise NotADirectoryError(f"Taxonomy directory not found: {tax_dir}")
            nodes_path = os.path.join(tax_dir, "nodes.dmp")
            names_path = os.path.join(tax_dir, "names.dmp")
            
        if nodes_path and names_path:
            if not os.path.exists(nodes_path):
                raise FileNotFoundError(f"Taxonomy node file not found: {nodes_path}")
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"Taxonomy names file not found: {names_path}")
            self.build_from_dmp(nodes_path, names_path)
        elif nodes_path or names_path:
            raise ValueError("Both 'nodes' and 'names' must be provided (or 'tax_dir').")

    def build_from_dmp(self, nodes: str, names: str) -> None:
        """
        Parses NCBI DMP files and builds the optimized vectorized internal structure.
        
        This process involves:
        1. Parsing scientific and common names into a Polars search index.
        2. Detecting the top rank (superkingdom vs domain).
        3. Integrity Check: Validating tree structure (orphans, roots, self-parenting).
        4. Creating a dense internal mapping (0 to N-1) for all TaxIDs.
        5. Calculating node depths and Euler Tour timestamps for instant clade queries.
        6. Integrity Check: Detecting cycles during depth calculation.
        7. Pre-calculating ancestors for all canonical ranks.
        8. Integrity Check: Detecting redundant canonical ranks in lineages.
        
        Args:
            nodes: Path to NCBI nodes.dmp.
            names: Path to NCBI names.dmp.
            
        Raises:
            ValueError: If both 'superkingdom' and 'domain' ranks are found.
            TaxonomyIntegrityError: If the tree is corrupt (cycles, orphans, etc).
        """
        self._source_nodes = os.path.abspath(nodes)
        self._source_names = os.path.abspath(names)
        self._build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting taxonomy build at {self._build_time}...")
        
        # 1. Parse Names
        logger.info(f"Parsing names from {names}...")
        scientific_names = {}
        common_names = {}
        search_data = [] # List of (name, tax_id)
        
        with open(names, 'r') as f:
            for name_line in f:
                parts = name_line.split('|')
                name_type = parts[3].strip()
                
                # Only care about scientific and genbank common names for now
                if name_type not in ['scientific name', 'genbank common name']:
                    continue
                
                tax_id = int(parts[0].strip())
                name_txt = parts[1].strip()
                
                if name_type == 'scientific name':
                    scientific_names[tax_id] = name_txt
                elif name_type == 'genbank common name':
                    common_names[tax_id] = name_txt
                
                search_data.append({"name": name_txt, "tax_id": tax_id})
        
        # Build search index
        self._search_index = pl.DataFrame(search_data).sort("name")
        
        # 2. Parse Nodes and initial parent structure
        logger.info(f"Parsing nodes from {nodes}...")
        temp_parents = {}
        temp_ranks = {}
        all_ranks = set()
        
        with open(nodes, 'r') as f:
            for line in f:
                parts = line.split('|')
                tax_id = int(parts[0].strip())
                parent_id = int(parts[1].strip())
                rank = parts[2].strip()
                
                temp_parents[tax_id] = parent_id
                temp_ranks[tax_id] = rank
                all_ranks.add(rank)
        
        # 2.0 Integrity Check: Orphans and Multiple Roots
        if 1 not in temp_parents:
            raise TaxonomyIntegrityError("TaxID 1 (root) is missing from the taxonomy nodes.")
            
        for tid, pid in temp_parents.items():
            if pid not in temp_parents:
                raise TaxonomyIntegrityError(f"Node {tid} has parent {pid} which is missing from the taxonomy.")
            if pid == tid and tid != 1:
                raise TaxonomyIntegrityError(f"Self-parenting loop detected at node {tid}. Only TaxID 1 should be its own parent.")
        
        # 2.1 Detect top rank (superkingdom vs domain)
        has_sk = 'superkingdom' in all_ranks
        has_dm = 'domain' in all_ranks
        if has_sk and has_dm:
            raise ValueError("Found both 'superkingdom' and 'domain' ranks. The taxonomy must use only one as the top rank.")
        self.top_rank = 'superkingdom' if has_sk else 'domain'
        logger.info(f"Detected top rank: {self.top_rank}")

        # 3. Create dense mapping
        logger.info("Creating dense mapping and vectorized arrays...")
        sorted_tax_ids = sorted(temp_parents.keys())
        num_nodes = len(sorted_tax_ids)
        self._index_to_id = np.array(sorted_tax_ids, dtype=np.int32)
        
        # Mapping rank names to indices
        self.rank_names = sorted(list(all_ranks))
        rank_to_idx = {r: i for i, r in enumerate(self.rank_names)}
        
        self.parents = np.zeros(num_nodes, dtype=np.int32)
        self.ranks = np.zeros(num_nodes, dtype=np.uint8)
        
        # Temporary dict for building parent connections (will be discarded)
        id_to_index_temp = {tid: i for i, tid in enumerate(sorted_tax_ids)}
        
        for tid, i in id_to_index_temp.items():
            parent_id = temp_parents[tid]
            # Handle root (1) which is its own parent in NCBI
            if tid == 1:
                self.parents[i] = i
            else:
                self.parents[i] = id_to_index_temp[parent_id]
            
            self.ranks[i] = rank_to_idx[temp_ranks[tid]]

        # Populate names aligned with indices
        logger.info("Aligning names and ranks...")
        sci_names_list = [scientific_names.get(tid, f"Unknown_{tid}") for tid in sorted_tax_ids]
        com_names_list = [common_names.get(tid) for tid in sorted_tax_ids]
        self._scientific_names = pl.Series("scientific_name", sci_names_list)
        self._common_names = pl.Series("common_name", com_names_list)

        # 4. Calculate depths
        logger.info("Calculating node depths...")
        self.depths = np.zeros(num_nodes, dtype=np.int32)
        for i in range(num_nodes):
            self._calculate_depth(i)
            
        # 5. Build Euler Tour for clade queries
        self._build_euler_tour()

        # 6. Pre-calculate canonical rank maps
        self._build_canonical_maps()
        
        # 7. Prepare caches for vectorized lookups
        self._prepare_vectorized_caches()
        
        logger.info("Taxonomy build complete.")

    def _prepare_vectorized_caches(self) -> None:
        """Initializes caches used for high-performance vectorized lookups."""
        logger.info("Preparing vectorized lookup caches...")
        # Scientific names lookup (aligned with dense internal index + 1 for "Unknown")
        self._sci_names_lookup = self._scientific_names.append(pl.Series([None]))
        
        # Rank names lookup
        self._rank_names_series = pl.Series(self.rank_names).append(pl.Series(["unclassified"]))
        
        # Ranks extended with a pointer to "unclassified" for unknown nodes
        self._ranks_extended = np.append(self.ranks, [len(self.rank_names)]).astype(np.int32)

    def _build_canonical_maps(self) -> None:
        """Pre-calculates canonical rank ancestors for all nodes."""
        logger.info("Pre-calculating canonical rank maps...")
        num_nodes = len(self._index_to_id)
        
        # Identify all canonical ranks to track
        canonical_columns = [self.top_rank] + [r for r in CANONICAL_RANKS if r not in ['superkingdom', 'domain']]
        
        # Initialize maps with -1 (meaning no ancestor at that rank)
        self.canonical_maps = {rank: np.full(num_nodes, -1, dtype=np.int32) for rank in canonical_columns}
        
        # Sort nodes by depth to ensure parents are processed before children
        for i in range(num_nodes):
            curr_idx = i
            root_idx = 0 # TaxID 1 is always the first in sorted_tax_ids
            seen_ranks = set()
            while True:
                rank_name = self.rank_names[self.ranks[curr_idx]]
                
                # Normalize superkingdom/domain based on detected top_rank
                mapped_rank = rank_name
                if rank_name in ['superkingdom', 'domain']:
                    mapped_rank = self.top_rank
                
                if mapped_rank in self.canonical_maps:
                    if mapped_rank in seen_ranks:
                        raise TaxonomyIntegrityError(
                            f"Multiple nodes of canonical rank '{mapped_rank}' found in lineage of node {self._index_to_id[i]}. "
                            f"Ancestors: {self._index_to_id[curr_idx]} and another node."
                        )
                    seen_ranks.add(mapped_rank)
                    self.canonical_maps[mapped_rank][i] = curr_idx
                
                if curr_idx == root_idx:
                    break
                curr_idx = self.parents[curr_idx]

    def _calculate_depth(self, index: int, path: Optional[Set[int]] = None) -> int:
        """Recursive depth calculation with memoization and cycle detection."""
        if index == 0: # TaxID 1 is always index 0
            return 0
        if self.depths[index] != 0:
            return self.depths[index]
            
        if path is None:
            path = set()
            
        if index in path:
            raise TaxonomyIntegrityError(f"Cycle detected involving TaxID {self._index_to_id[index]}")
            
        path.add(index)
        d = self._calculate_depth(self.parents[index], path) + 1
        path.remove(index)
        self.depths[index] = d
        return d

    def _build_euler_tour(self) -> None:
        """Assigns entry/exit times to enable instant clade queries."""
        logger.info("Building Euler Tour index for clade queries...")
        num_nodes = len(self._index_to_id)
        self.entry_times = np.zeros(num_nodes, dtype=np.int32)
        self.exit_times = np.zeros(num_nodes, dtype=np.int32)
        
        # Build adjacency list (children)
        children = [[] for _ in range(num_nodes)]
        root_idx = 0 # TaxID 1
        for i, p in enumerate(self.parents):
            if i != root_idx:
                children[p].append(i)
        
        timer = 0
        stack = [(root_idx, False)] # (index, is_processed)
        
        while stack:
            idx, processed = stack.pop()
            if not processed:
                self.entry_times[idx] = timer
                timer += 1
                stack.append((idx, True))
                for child in reversed(children[idx]):
                    stack.append((child, False))
            else:
                self.exit_times[idx] = timer - 1

    def _get_index(self, tax_id: int) -> int:
        """
        Returns the dense internal index for a given NCBI TaxID.
        
        Args:
            tax_id: The NCBI TaxID to look up.
            
        Returns:
            The 0-based internal index, or -1 if the TaxID is not in the tree.
        """
        idx = np.searchsorted(self._index_to_id, tax_id)
        if idx < len(self._index_to_id) and self._index_to_id[idx] == tax_id:
            return int(idx)
        return -1
    
    def _get_indices(self, tax_ids: np.ndarray) -> np.ndarray:
        """
        Vectorized lookup of internal indices for an array of NCBI TaxIDs.
        
        Args:
            tax_ids: A NumPy array of NCBI TaxIDs.
            
        Returns:
            A NumPy array of internal indices, with -1 for missing TaxIDs.
        """
        indices = np.searchsorted(self._index_to_id, tax_ids)
        # Handle out of bounds
        mask = indices < len(self._index_to_id)
        # Check for actual equality
        valid = np.zeros(len(tax_ids), dtype=bool)
        valid[mask] = self._index_to_id[indices[mask]] == tax_ids[mask]
        return np.where(valid, indices, -1)

    def get_lineage(self, tax_id: int, strict: bool = True) -> List[int]:
        """
        Returns the full path of TaxIDs from the root to the given TaxID.
        
        Args:
            tax_id: The NCBI TaxID to trace.
            strict: If True, raises TaxIDNotFoundError if the ID is not in the tree.
            
        Returns:
            A list of TaxIDs starting from root (1) down to the query ID.
            Returns an empty list if the TaxID is not found and strict=False.
            
        Example:
            >>> tree.get_lineage(562) # E. coli
            [1, 2, 1224, 1236, 91347, 543, 561, 562]
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx == -1:
            if strict:
                raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        lineage = []
        root_idx = 0
        
        while True:
            lineage.append(int(self._index_to_id[idx]))
            if idx == root_idx:
                break
            idx = self.parents[idx]
            
        return lineage[::-1]

    def get_name(self, tax_id: int, strict: bool = True) -> Optional[str]:
        """
        Returns the scientific name for a given NCBI TaxID.
        
        Args:
            tax_id: The NCBI TaxID.
            strict: If True, raises TaxIDNotFoundError if the ID is not in the tree.
            
        Returns:
            The scientific name string, or None if not found and strict=False.
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx != -1:
            return self._scientific_names[idx]
        
        if strict:
            raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
        return None

    def get_common_name(self, tax_id: int, strict: bool = True) -> Optional[str]:
        """
        Returns the GenBank common name for a given NCBI TaxID, if available.
        
        Args:
            tax_id: The NCBI TaxID.
            strict: If True, raises TaxIDNotFoundError if the ID is not in the tree.
            
        Returns:
            The common name string, or None if not available or if strict=False and ID is missing.
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx != -1:
            return self._common_names[idx]
        
        if strict:
            raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
        return None

    def get_rank(self, tax_id: int, strict: bool = True) -> Optional[str]:
        """
        Returns the taxonomic rank for a given NCBI TaxID.
        
        Args:
            tax_id: The NCBI TaxID.
            strict: If True, raises TaxIDNotFoundError if the ID is not in the tree.
            
        Returns:
            The rank name (e.g., 'species', 'genus'), or None if not found and strict=False.
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx != -1:
            return self.rank_names[self.ranks[idx]]
        
        if strict:
            raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
        return None

    def search_name(self, query: str, fuzzy: bool = False, limit: int = 10, score_cutoff: float = 60.0) -> pl.DataFrame:
        """
        Searches for TaxIDs by name using exact or fuzzy matching.
        
        Args:
            query: The name string to search for.
            fuzzy: If True, uses RapidFuzz for approximate matching.
            limit: Maximum number of candidates to return (fuzzy only).
            score_cutoff: Minimum similarity score (0-100) for fuzzy matches.
            
        Returns:
            A Polars DataFrame with columns: ['tax_id', 'matched_name', 'scientific_name', 'rank', 'score'].
            
        Example:
            >>> tree.search_name("Escherchia", fuzzy=True, limit=1)
            shape: (1, 5)
            ┌────────┬──────────────┬──────────────────┬────────┬───────┐
            │ tax_id ┆ matched_name ┆ scientific_name  ┆ rank   ┆ score │
            ╞════════╪══════════════╪══════════════════╪════════╪═══════╡
            │ 561    ┆ Escherichia  ┆ Escherichia      ┆ genus  ┆ 92.0  │
            └────────┴──────────────┴──────────────────┴────────┴───────┘
        """
        if not fuzzy:
            matches = self._search_index.filter(pl.col("name") == query)
            if matches.is_empty():
                return pl.DataFrame(schema=["tax_id", "name", "rank", "score"])
            
            # Vectorized rank lookup for matches
            tids = matches["tax_id"].to_numpy()
            indices = self._get_indices(tids)
            ranks = [self.rank_names[self.ranks[i]] if i != -1 else "unknown" for i in indices]
            
            return matches.with_columns([
                pl.Series("rank", ranks),
                pl.lit(100.0).alias("score")
            ])

        # Fuzzy matching path
        unique_names = self._search_index["name"].unique().to_list()
            
        # rapidfuzz extract
        matches = process.extract(
            query, 
            unique_names, 
            scorer=fuzz.WRatio, 
            limit=limit, 
            processor=utils.default_process,
            score_cutoff=score_cutoff
        )
        
        data = []
        for match_str, score, _ in matches:
            # Find all TaxIDs associated with this name
            tids = self._search_index.filter(pl.col("name") == match_str)["tax_id"].to_list()
            for tid in tids:
                idx = self._get_index(tid)
                rank = self.rank_names[self.ranks[idx]] if idx != -1 else "unknown"
                
                # Smart Ranking: Boost scores for canonical ranks
                rank_boost = 0.0
                if rank in CANONICAL_RANKS:
                    rank_boost = 2.0
                
                data.append({
                    "tax_id": tid,
                    "matched_name": match_str,
                    "scientific_name": self.get_name(tid),
                    "rank": rank,
                    "score": score + rank_boost
                })
        
        if not data:
            return pl.DataFrame(schema=["tax_id", "matched_name", "scientific_name", "rank", "score"])
            
        return pl.DataFrame(data).sort("score", descending=True)

    def get_clade(self, tax_id: int, strict: bool = True) -> List[int]:
        """
        Returns all TaxIDs in the clade (descendants) rooted at the given TaxID.
        
        Uses Euler Tour range indexing for O(1) identification of descendants.
        
        Args:
            tax_id: The root NCBI TaxID of the clade.
            strict: If True, raises TaxIDNotFoundError if the ID is not in the tree.
            
        Returns:
            A list of NCBI TaxIDs belonging to the clade.
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx == -1:
            if strict:
                raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        entry = self.entry_times[idx]
        exit = self.exit_times[idx]
        
        mask = (self.entry_times >= entry) & (self.entry_times <= exit)
        return self._index_to_id[mask].astype(int).tolist()

    def get_clade_at_rank(self, tax_id: int, rank_name: str, strict: bool = True) -> List[int]:
        """
        Returns all descendants of a specific rank within the clade rooted at tax_id.
        
        Args:
            tax_id: The root NCBI TaxID of the clade.
            rank_name: The target rank name (e.g., 'species').
            strict: If True, raises TaxIDNotFoundError if the tax_id is not in the tree.
            
        Returns:
            A list of NCBI TaxIDs of the target rank within the clade.
            
        Example:
            >>> tree.get_clade_at_rank(2, 'phylum') # All phyla in Bacteria
            [1224, 201174, ...]
        """
        if not isinstance(tax_id, (int, np.integer)):
            raise TypeError(f"TaxID must be an integer, got {type(tax_id).__name__}")
            
        idx = self._get_index(tax_id)
        if idx == -1:
            if strict:
                raise TaxIDNotFoundError(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        try:
            target_rank_idx = self.rank_names.index(rank_name)
        except ValueError:
            logger.warning(f"Rank '{rank_name}' not found in taxonomy. Available ranks: {self.rank_names}")
            return []
        
        entry = self.entry_times[idx]
        exit = self.exit_times[idx]
        
        mask = (self.entry_times >= entry) & (self.entry_times <= exit) & (self.ranks == target_rank_idx)
        return self._index_to_id[mask].astype(int).tolist()

    def get_lca(self, tax_id_1: int, tax_id_2: int, strict: bool = True) -> Optional[int]:
        """
        Finds the Lowest Common Ancestor (LCA) of two NCBI TaxIDs.
        
        Uses Hyper-Vectorized binary lifting for O(log Depth) performance.
        
        Args:
            tax_id_1: First NCBI TaxID.
            tax_id_2: Second NCBI TaxID.
            strict: If True, raises TaxIDNotFoundError if either ID is missing.
            
        Returns:
            The NCBI TaxID of the LCA. Returns None if one or both IDs are missing and strict=False.
        """
        if not isinstance(tax_id_1, (int, np.integer)):
            raise TypeError(f"tax_id_1 must be an integer, got {type(tax_id_1).__name__}")
        if not isinstance(tax_id_2, (int, np.integer)):
            raise TypeError(f"tax_id_2 must be an integer, got {type(tax_id_2).__name__}")
            
        idx1 = self._get_index(tax_id_1)
        idx2 = self._get_index(tax_id_2)
        
        if idx1 == -1 or idx2 == -1:
            if strict:
                missing = tax_id_1 if idx1 == -1 else tax_id_2
                raise TaxIDNotFoundError(f"TaxID {missing} not found in taxonomy tree.")
            return None
            
        self._ensure_up_table()
        up_table = self._up_table
        assert up_table is not None, "up_table must be initialized"
        
        if self.depths[idx1] < self.depths[idx2]:
            idx1, idx2 = idx2, idx1
        
        diff = self.depths[idx1] - self.depths[idx2]
        max_log = up_table.shape[0]
        
        for i in range(max_log):
            if (diff >> i) & 1:
                idx1 = up_table[i, idx1]
                
        if idx1 == idx2:
            return int(self._index_to_id[idx1])
            
        for i in reversed(range(max_log)):
            up1 = up_table[i, idx1]
            up2 = up_table[i, idx2]
            if up1 != up2:
                idx1 = up1
                idx2 = up2
                
        return int(self._index_to_id[self.parents[idx1]])

    def get_distance(self, tax_id_1: int, tax_id_2: int, strict: bool = True) -> Optional[int]:
        """
        Calculates the distance (number of edges) between two NCBI TaxIDs.
        
        Args:
            tax_id_1: First NCBI TaxID.
            tax_id_2: Second NCBI TaxID.
            strict: If True, raises TaxIDNotFoundError if either ID is missing.
            
        Returns:
            The number of edges between the nodes. Returns None if nodes are missing and strict=False.
        """
        if not isinstance(tax_id_1, (int, np.integer)):
            raise TypeError(f"tax_id_1 must be an integer, got {type(tax_id_1).__name__}")
        if not isinstance(tax_id_2, (int, np.integer)):
            raise TypeError(f"tax_id_2 must be an integer, got {type(tax_id_2).__name__}")
            
        lca_id = self.get_lca(tax_id_1, tax_id_2, strict=strict)
        if lca_id is None:
            return None
            
        idx1 = self._get_index(tax_id_1)
        idx2 = self._get_index(tax_id_2)
        idx_lca = self._get_index(lca_id)
        
        return int(self.depths[idx1] + self.depths[idx2] - 2 * self.depths[idx_lca])

    def get_lca_batch(self, ids1: Union[List[int], np.ndarray], ids2: Union[List[int], np.ndarray], strict: bool = True) -> np.ndarray:
        """
        Calculates Lowest Common Ancestor for arrays of NCBI TaxIDs.
        
        Hyper-vectorized implementation using transposed binary lifting.
        Capable of resolving millions of pairs per second.
        
        Args:
            ids1: First array of NCBI TaxIDs.
            ids2: Second array of NCBI TaxIDs.
            strict: If True, raises TaxIDNotFoundError if any ID is missing from the tree.
            
        Returns:
            A NumPy array of LCA TaxIDs. Missing IDs result in -1 if strict=False.
        """
        if not isinstance(ids1, (list, np.ndarray)):
            raise TypeError(f"ids1 must be a list or numpy array, got {type(ids1).__name__}")
        if not isinstance(ids2, (list, np.ndarray)):
            raise TypeError(f"ids2 must be a list or numpy array, got {type(ids2).__name__}")
            
        ids1_arr = np.array(ids1, dtype=np.int32)
        ids2_arr = np.array(ids2, dtype=np.int32)
        
        if ids1_arr.shape != ids2_arr.shape:
            raise ValueError("Input arrays must have the same shape.")
            
        self._ensure_up_table()
        up_table = self._up_table
        assert up_table is not None, "up_table must be initialized"
        
        idx1 = self._get_indices(ids1_arr)
        idx2 = self._get_indices(ids2_arr)
        
        if strict:
            missing1 = ids1_arr[idx1 == -1]
            missing2 = ids2_arr[idx2 == -1]
            if len(missing1) > 0 or len(missing2) > 0:
                first_missing = missing1[0] if len(missing1) > 0 else missing2[0]
                raise TaxIDNotFoundError(f"TaxID {first_missing} (and possibly others) not found in taxonomy tree.")
        
        # Handle missing IDs by pointing to root (index 0)
        valid_mask = (idx1 != -1) & (idx2 != -1)
        s_idx1 = np.where(valid_mask, idx1, 0)
        s_idx2 = np.where(valid_mask, idx2, 0)
        
        # 1. Bring both nodes to the same depth
        d1 = self.depths[s_idx1]
        d2 = self.depths[s_idx2]
        
        # Ensure s_idx1 is the deeper one
        swap = d1 < d2
        s_idx1[swap], s_idx2[swap] = s_idx2[swap], s_idx1[swap]
        
        diff = np.abs(d1 - d2)
        max_log = up_table.shape[0]
        
        for i in range(max_log):
            mask = (diff >> i) & 1 == 1
            if np.any(mask):
                s_idx1[mask] = up_table[i, s_idx1[mask]]
            
        # 2. Binary search for the LCA
        lca_indices = s_idx1.copy()
        not_same = s_idx1 != s_idx2
        
        if np.any(not_same):
            sub1 = s_idx1[not_same]
            sub2 = s_idx2[not_same]
            
            for i in reversed(range(max_log)):
                up1 = up_table[i, sub1]
                up2 = up_table[i, sub2]
                
                diff_up = up1 != up2
                sub1[diff_up] = up1[diff_up]
                sub2[diff_up] = up2[diff_up]
            
            lca_indices[not_same] = self.parents[sub1]
            
        results = self._index_to_id[lca_indices]
        # Return -1 for pairs involving missing IDs
        results[~valid_mask] = -1
        return results

    def get_distance_batch(self, ids1: Union[List[int], np.ndarray], ids2: Union[List[int], np.ndarray], strict: bool = True) -> np.ndarray:
        """
        Vectorized distance calculation for arrays of NCBI TaxIDs.
        
        Args:
            ids1: First array of NCBI TaxIDs.
            ids2: Second array of NCBI TaxIDs.
            strict: If True, raises TaxIDNotFoundError if any ID is missing from the tree.
            
        Returns:
            A NumPy array of edge distances. Missing IDs result in -1 if strict=False.
        """
        if not isinstance(ids1, (list, np.ndarray)):
            raise TypeError(f"ids1 must be a list or numpy array, got {type(ids1).__name__}")
        if not isinstance(ids2, (list, np.ndarray)):
            raise TypeError(f"ids2 must be a list or numpy array, got {type(ids2).__name__}")
            
        ids1_arr = np.array(ids1, dtype=np.int32)
        ids2_arr = np.array(ids2, dtype=np.int32)
        
        lca_ids = self.get_lca_batch(ids1_arr, ids2_arr, strict=strict)
        
        idx1 = self._get_indices(ids1_arr)
        idx2 = self._get_indices(ids2_arr)
        idx_lca = self._get_indices(lca_ids)
        
        # Mask invalid lookups
        valid = (idx1 != -1) & (idx2 != -1) & (idx_lca != -1)
        
        dists = np.full(len(ids1_arr), -1, dtype=np.int32)
        if np.any(valid):
            v1, v2, vl = idx1[valid], idx2[valid], idx_lca[valid]
            dists[valid] = self.depths[v1] + self.depths[v2] - 2 * self.depths[vl]
            
        return dists

    def annotate(self, tax_ids: Union[int, List[int], np.ndarray], strict: bool = True) -> pl.DataFrame:
        """
        Massively annotates one or more TaxIDs with scientific names and canonical ranks.
        
        Extremely efficient for large tables (e.g., millions of rows) using 
        Polars vectorized 'gather' and pre-calculated canonical rank maps.
        
        Args:
            tax_ids: A single NCBI TaxID (int) or an iterable (list, NumPy array).
            strict: If True, raises TaxIDNotFoundError if any ID is missing from the tree.
            
        Returns:
            A Polars DataFrame containing columns for each canonical rank (prefixed with 't_'),
            plus 't_id', 't_scientific_name' and 't_rank'.
            
        Example:
            >>> tree.annotate(562) # Single ID works
            >>> tree.annotate([9606, 562]) # Batch works
        """
        if not isinstance(tax_ids, (int, np.integer, list, np.ndarray)):
            raise TypeError(f"tax_ids must be an integer, list, or numpy array, got {type(tax_ids).__name__}")
            
        if isinstance(tax_ids, (int, np.integer)):
            ids_arr = np.array([int(tax_ids)], dtype=np.int32)
        else:
            # tax_ids must be list or np.ndarray here
            ids_arr = np.array(tax_ids, dtype=np.int32)
            
        indices = self._get_indices(ids_arr)
        
        if strict:
            missing = ids_arr[indices == -1]
            if len(missing) > 0:
                raise TaxIDNotFoundError(f"TaxID {missing[0]} (and possibly others) not found in taxonomy tree.")

        logger.info(f"Annotating {len(ids_arr)} taxa...")
        canonical_columns = [self.top_rank] + [r for r in CANONICAL_RANKS if r not in ['superkingdom', 'domain']]
        
        valid_mask = indices != -1
        
        # dummy_idx points to the "Unknown/None" entry at the end of the lookup series
        dummy_idx = len(self._index_to_id)
        safe_indices = np.where(valid_mask, indices, dummy_idx)
        
        # Ensure caches are ready
        if self._sci_names_lookup is None:
            self._prepare_vectorized_caches()
        
        sci_names_lookup = self._sci_names_lookup
        rank_names_series = self._rank_names_series
        ranks_extended = self._ranks_extended
        
        assert sci_names_lookup is not None
        assert rank_names_series is not None
        assert ranks_extended is not None
            
        df_dict = {"t_id": ids_arr}
        
        for rank in canonical_columns:
            # canonical_maps now store internal indices
            ancestor_indices = np.full(len(ids_arr), -1, dtype=np.int32)
            # Map input tax_ids to their ancestor's internal index
            ancestor_indices[valid_mask] = self.canonical_maps[rank][indices[valid_mask]]
            
            # Use dummy_idx for missing ancestors
            safe_anc_indices = np.where(ancestor_indices != -1, ancestor_indices, dummy_idx)
            
            # Vectorized gather from Polars
            df_dict[f"t_{rank}"] = sci_names_lookup.gather(safe_anc_indices.astype(np.int32))

        # Scientific name for the input TaxID
        df_dict["t_scientific_name"] = sci_names_lookup.gather(safe_indices.astype(np.int32))
        
        # Rank for the input TaxID
        target_rank_indices = ranks_extended[safe_indices]
        df_dict["t_rank"] = rank_names_series.gather(target_rank_indices.astype(np.int32))
        
        df = pl.DataFrame(df_dict)
        final_order = ['t_id'] + [f"t_{rank}" for rank in canonical_columns] + ['t_scientific_name', 't_rank']
        return df.select(final_order)

    @property
    def available_ranks(self) -> List[str]:
        """Returns a sorted list of all taxonomic ranks present in this tree."""
        return sorted(self.rank_names)

    @property
    def summary(self) -> Dict[str, Any]:
        """Returns a summary dictionary of the tree's metadata and provenance."""
        return {
            "node_count": len(self._index_to_id),
            "top_rank": self.top_rank,
            "build_time": self._build_time,
            "source_nodes": self._source_nodes,
            "source_names": self._source_names,
            "package_version": __version__,
            "max_depth": int(np.max(self.depths)) if len(self.depths) > 0 else 0,
            "ranks_present": len(self.rank_names)
        }

    def _ensure_up_table(self) -> None:
        """
        Lazy initialization of the binary lifting table.
        
        Constructs a hyper-vectorized table of shape (max_log, num_nodes) 
        where up_table[j, i] is the 2^j-th ancestor of node i.
        """
        if self._up_table is not None:
            return
            
        logger.info("Initializing binary lifting table (Hyper-Vectorized)...")
        num_nodes = len(self._index_to_id)
        max_log = int(np.ceil(np.log2(np.max(self.depths) + 1)))
        
        # Shape: (max_log, num_nodes) - optimized for contiguous column access
        up_table = np.zeros((max_log, num_nodes), dtype=np.int32)
        self._up_table = up_table
        
        # Power 2^0 is just the parents
        up_table[0, :] = self.parents
        
        # Power 2^j = 2^{j-1} jump from the 2^{j-1} ancestor
        # Fully vectorized initialization
        for j in range(1, max_log):
            prev_ancestors = up_table[j-1, :]
            up_table[j, :] = up_table[j-1, prev_ancestors]

    def save(self, directory: str) -> None:
        """
        Saves the vectorized tree and metadata to a directory for fast loading.
        
        Uses NumPy .npy format for numerical arrays and Apache Arrow IPC for 
        Polars string stores, ensuring nearly instantaneous zero-copy loading.
        
        Args:
            directory: Path to the cache directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        logger.info(f"Saving binary cache to {directory}...")
        np.save(os.path.join(directory, "index_to_id.npy"), self._index_to_id)
        np.save(os.path.join(directory, "parents.npy"), self.parents)
        np.save(os.path.join(directory, "depths.npy"), self.depths)
        np.save(os.path.join(directory, "ranks.npy"), self.ranks)
        np.save(os.path.join(directory, "entry_times.npy"), self.entry_times)
        np.save(os.path.join(directory, "exit_times.npy"), self.exit_times)
        
        # Save Polars metadata
        self._scientific_names.to_frame().write_ipc(os.path.join(directory, "scientific_names.ipc"))
        self._common_names.to_frame().write_ipc(os.path.join(directory, "common_names.ipc"))
        self._search_index.write_ipc(os.path.join(directory, "search_index.ipc"))

        maps_dir = os.path.join(directory, "canonical_maps")
        if not os.path.exists(maps_dir):
            os.makedirs(maps_dir)
        for rank, arr in self.canonical_maps.items():
            np.save(os.path.join(maps_dir, f"{rank}.npy"), arr)
            
        import pickle
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                "rank_names": self.rank_names,
                "top_rank": self.top_rank,
                "provenance": {
                    "build_time": self._build_time,
                    "source_nodes": self._source_nodes,
                    "source_names": self._source_names,
                    "package_version": __version__,
                    "node_count": len(self._index_to_id),
                    "max_depth": int(np.max(self.depths))
                }
            }, f)

    @classmethod
    def load(cls, directory: str) -> 'JolTree':
        """
        Loads a vectorized tree from a binary cache directory.
        
        Validates the cache version to ensure compatibility with the current 
        package version.
        
        Args:
            directory: Path to the binary cache directory.
            
        Returns:
            An initialized JolTree object.
            
        Raises:
            RuntimeError: If the cache version is incompatible.
        """
        logger.info(f"Loading binary cache from {directory}...")
        
        import pickle
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            meta = pickle.load(f)
            prov = meta.get("provenance", {})
            saved_version = prov.get("package_version", "unknown")
            def version_to_tuple(v):
                try:
                    return tuple(map(int, v.split('.')))
                except (ValueError, AttributeError):
                    return (0, 0, 0)
            if version_to_tuple(saved_version) < version_to_tuple(MINIMUM_CACHE_VERSION):
                raise RuntimeError(
                    f"Incompatible taxonomy cache. Saved version: {saved_version}, "
                    f"Minimum required: {MINIMUM_CACHE_VERSION}. Please rebuild with build_from_dmp()."
                )

            tree = cls()
            tree.rank_names = meta["rank_names"]
            tree.top_rank = meta.get("top_rank", "domain")
            tree._build_time = prov.get("build_time")
            tree._source_nodes = prov.get("source_nodes")
            tree._source_names = prov.get("source_names")

        tree._index_to_id = np.load(os.path.join(directory, "index_to_id.npy"))
        tree.parents = np.load(os.path.join(directory, "parents.npy"))
        tree.depths = np.load(os.path.join(directory, "depths.npy"))
        tree.ranks = np.load(os.path.join(directory, "ranks.npy"))
        tree.entry_times = np.load(os.path.join(directory, "entry_times.npy"))
        tree.exit_times = np.load(os.path.join(directory, "exit_times.npy"))
        
        # Load Polars metadata
        tree._scientific_names = pl.read_ipc(os.path.join(directory, "scientific_names.ipc"))["scientific_name"]
        tree._common_names = pl.read_ipc(os.path.join(directory, "common_names.ipc"))["common_name"]
        tree._search_index = pl.read_ipc(os.path.join(directory, "search_index.ipc"))

        maps_dir = os.path.join(directory, "canonical_maps")
        if os.path.exists(maps_dir):
            for filename in os.listdir(maps_dir):
                if filename.endswith(".npy"):
                    rank = filename[:-4]
                    tree.canonical_maps[rank] = np.load(os.path.join(maps_dir, filename))
        
        # Re-initialize vectorized caches
        tree._prepare_vectorized_caches()
        
        logger.info("Loaded taxonomy cache:")
        logger.info(f"  Version:       {saved_version}")
        logger.info(f"  Build time:    {tree._build_time}")
        logger.info(f"  Node count:    {prov.get('node_count', 'Unknown'):,}")
        logger.info(f"  Top rank:      {tree.top_rank}")
        return tree
