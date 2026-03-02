#!/usr/bin/env python3
"""
taxatree/tree.py
Implementation of a high-performance, vectorized taxonomy tree.
"""

__version__ = "0.1.0"

import logging
import os
import datetime
from typing import Dict, List, Optional, Set, Union, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd
import polars as pl

# Set up logging for the module
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d [%H:%M:%S]'
)
logger = logging.getLogger(__name__)

# Standard canonical ranks in order (highest to lowest)
# Including both superkingdom and domain for compatibility with pre/post-2025 taxonomies
CANONICAL_RANKS = [
    'superkingdom', 'domain', 'kingdom', 'phylum', 
    'class', 'order', 'family', 'genus', 'species'
]

# Mapping rank names to standard Kraken-style codes
RANK_TO_CODE = {
    'superkingdom': 'D',
    'domain': 'D',
    'kingdom': 'K',
    'phylum': 'P',
    'class': 'C',
    'order': 'O',
    'family': 'F',
    'genus': 'G',
    'species': 'S'
}

class TaxonomyTree:
    """
    A high-performance taxonomy representation using vectorized arrays.
    
    This class replaces traditional object-oriented trees with contiguous 
    NumPy arrays for lightning-fast lookups, traversals, and mass annotations.
    """

    def __init__(self, nodes_file: Optional[str] = None, names_file: Optional[str] = None):
        """
        Initialize the taxonomy tree. If files are provided, it builds from DMP files.
        Otherwise, it can be loaded from a binary cache using `load()`.
        
        Args:
            nodes_file: Path to NCBI nodes.dmp
            names_file: Path to NCBI names.dmp
        """
        # Internal mapping from original TaxID to dense 0-based index
        self._id_to_index: Dict[int, int] = {}
        self._index_to_id: np.ndarray = np.array([], dtype=np.int32)
        
        # Primary arrays (indexed by the dense internal index)
        self.parents: np.ndarray = np.array([], dtype=np.int32)
        self.depths: np.ndarray = np.array([], dtype=np.int32)
        self.ranks: np.ndarray = np.array([], dtype=np.uint8)
        
        # Metadata storage
        self.names: Dict[int, str] = {}
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
        
        # Pre-calculated canonical rank maps (dense internal index -> TaxID)
        # Dictionary mapping rank name to np.ndarray of shape (num_nodes,)
        self.canonical_maps: Dict[str, np.ndarray] = {}
        
        if nodes_file and names_file:
            self.build_from_dmp(nodes_file, names_file)

    def build_from_dmp(self, nodes_file: str, names_file: str) -> None:
        """
        Parses NCBI DMP files and builds the vectorized internal structure.
        
        Args:
            nodes_file: Path to NCBI nodes.dmp
            names_file: Path to NCBI names.dmp
        """
        self._source_nodes = os.path.abspath(nodes_file)
        self._source_names = os.path.abspath(names_file)
        self._build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting taxonomy build at {self._build_time}...")
        
        # 1. Parse Names
        logger.info(f"Parsing names from {names_file}...")
        raw_names = {}
        with open(names_file, 'r') as f:
            for name_line in f:
                if 'scientific name' in name_line:
                    parts = name_line.split('|')
                    tax_id = int(parts[0].strip())
                    name = parts[1].strip()
                    raw_names[tax_id] = name
        
        # 2. Parse Nodes and initial parent structure
        logger.info(f"Parsing nodes from {nodes_file}...")
        temp_parents = {}
        temp_ranks = {}
        all_ranks = set()
        
        with open(nodes_file, 'r') as f:
            for line in f:
                parts = line.split('|')
                tax_id = int(parts[0].strip())
                parent_id = int(parts[1].strip())
                rank = parts[2].strip()
                
                temp_parents[tax_id] = parent_id
                temp_ranks[tax_id] = rank
                all_ranks.add(rank)
        
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
        self._id_to_index = {tid: i for i, tid in enumerate(sorted_tax_ids)}
        self._index_to_id = np.array(sorted_tax_ids, dtype=np.int32)
        
        # Rank indexing
        self.rank_names = sorted(list(all_ranks))
        rank_to_idx = {r: i for i, r in enumerate(self.rank_names)}
        
        self.parents = np.zeros(num_nodes, dtype=np.int32)
        self.ranks = np.zeros(num_nodes, dtype=np.uint8)
        
        for tid, i in self._id_to_index.items():
            parent_id = temp_parents[tid]
            # Handle root (1) which is its own parent in NCBI
            if tid == 1:
                self.parents[i] = i
            else:
                self.parents[i] = self._id_to_index[parent_id]
            
            self.ranks[i] = rank_to_idx[temp_ranks[tid]]
            self.names[tid] = raw_names.get(tid, f"Unknown_{tid}")

        # 4. Calculate depths
        logger.info("Calculating node depths...")
        self.depths = np.zeros(num_nodes, dtype=np.int32)
        for i in range(num_nodes):
            self._calculate_depth(i)
            
        # 5. Build Euler Tour for clade queries
        self._build_euler_tour()

        # 6. Pre-calculate canonical rank maps
        self._build_canonical_maps()
        
        logger.info("Taxonomy build complete.")

    def _build_canonical_maps(self) -> None:
        """Pre-calculates canonical rank ancestors for all nodes."""
        logger.info("Pre-calculating canonical rank maps...")
        num_nodes = len(self._index_to_id)
        
        # Identify all canonical ranks to track
        canonical_columns = [self.top_rank] + [r for r in CANONICAL_RANKS if r not in ['superkingdom', 'domain']]
        
        # Initialize maps with -1 (meaning no ancestor at that rank)
        self.canonical_maps = {rank: np.full(num_nodes, -1, dtype=np.int32) for rank in canonical_columns}
        
        # Sort nodes by depth to ensure parents are processed before children
        # (Though we can also just walk up for each node, which is simpler to implement)
        for i in range(num_nodes):
            curr_idx = i
            root_idx = self._id_to_index[1]
            while True:
                rank_name = self.rank_names[self.ranks[curr_idx]]
                
                # Normalize superkingdom/domain based on detected top_rank
                mapped_rank = rank_name
                if rank_name in ['superkingdom', 'domain']:
                    mapped_rank = self.top_rank
                
                if mapped_rank in self.canonical_maps:
                    self.canonical_maps[mapped_rank][i] = self._index_to_id[curr_idx]
                
                if curr_idx == root_idx:
                    break
                curr_idx = self.parents[curr_idx]

    def _calculate_depth(self, index: int) -> int:
        """Recursive depth calculation with memoization."""
        if index == self._id_to_index[1]:
            return 0
        if self.depths[index] != 0:
            return self.depths[index]
        
        d = self._calculate_depth(self.parents[index]) + 1
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
        root_idx = self._id_to_index[1]
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
                # Add children in reverse to maintain some order consistency
                for child in reversed(children[idx]):
                    stack.append((child, False))
            else:
                self.exit_times[idx] = timer - 1

    def get_lineage(self, tax_id: int) -> List[int]:
        """Returns the full lineage from root to the given TaxID."""
        if tax_id not in self._id_to_index:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        idx = self._id_to_index[tax_id]
        lineage = []
        root_idx = self._id_to_index[1]
        
        while True:
            lineage.append(int(self._index_to_id[idx]))
            if idx == root_idx:
                break
            idx = self.parents[idx]
            
        return lineage[::-1]

    def get_name(self, tax_id: int) -> str:
        """Returns the scientific name of the given TaxID."""
        return self.names.get(tax_id, f"Unknown_{tax_id}")

    def get_rank(self, tax_id: int) -> str:
        """Returns the taxonomic rank of the given TaxID."""
        if tax_id not in self._id_to_index:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return "unknown"
        idx = self._id_to_index[tax_id]
        return self.rank_names[self.ranks[idx]]

    def get_clade(self, tax_id: int) -> List[int]:
        """Returns all TaxIDs in the clade rooted at the given TaxID."""
        if tax_id not in self._id_to_index:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        idx = self._id_to_index[tax_id]
        entry = self.entry_times[idx]
        exit = self.exit_times[idx]
        
        # Vectorized range check
        mask = (self.entry_times >= entry) & (self.entry_times <= exit)
        return self._index_to_id[mask].astype(int).tolist()

    def get_clade_at_rank(self, tax_id: int, rank_name: str) -> List[int]:
        """
        Returns all TaxIDs of a specific rank within the clade rooted at tax_id.
        Efficiently filters thousands of nodes using vectorized operations.
        """
        if tax_id not in self._id_to_index:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        # Determine the internal integer index for the target rank name
        try:
            target_rank_idx = self.rank_names.index(rank_name)
        except ValueError:
            logger.warning(f"Rank '{rank_name}' not found in taxonomy. Available ranks: {self.rank_names}")
            return []
        
        idx = self._id_to_index[tax_id]
        entry = self.entry_times[idx]
        exit = self.exit_times[idx]
        
        # Combined vectorized filter: Range check (clade) AND Rank match
        mask = (self.entry_times >= entry) & (self.entry_times <= exit) & (self.ranks == target_rank_idx)
        return self._index_to_id[mask].astype(int).tolist()

    def get_lca(self, tax_id_1: int, tax_id_2: int) -> int:
        """Finds the Lowest Common Ancestor using Binary Lifting."""
        if tax_id_1 not in self._id_to_index:
            logger.warning(f"TaxID {tax_id_1} not found in taxonomy tree.")
            return 1
        if tax_id_2 not in self._id_to_index:
            logger.warning(f"TaxID {tax_id_2} not found in taxonomy tree.")
            return 1
            
        self._ensure_up_table()
        idx1 = self._id_to_index[tax_id_1]
        idx2 = self._id_to_index[tax_id_2]
        
        # Bring both nodes to the same depth
        if self.depths[idx1] < self.depths[idx2]:
            idx1, idx2 = idx2, idx1
        
        diff = self.depths[idx1] - self.depths[idx2]
        for i in range(self._up_table.shape[1]):
            if (diff >> i) & 1:
                idx1 = self._up_table[idx1, i]
                
        if idx1 == idx2:
            return int(self._index_to_id[idx1])
            
        # Lift both until they share a parent
        for i in reversed(range(self._up_table.shape[1])):
            if self._up_table[idx1, i] != self._up_table[idx2, i]:
                idx1 = self._up_table[idx1, i]
                idx2 = self._up_table[idx2, i]
                
        return int(self._index_to_id[self.parents[idx1]])

    def get_distance(self, tax_id_1: int, tax_id_2: int) -> int:
        """Calculates distance (number of edges) between two TaxIDs."""
        lca_id = self.get_lca(tax_id_1, tax_id_2)
        
        idx1 = self._id_to_index[tax_id_1]
        idx2 = self._id_to_index[tax_id_2]
        idx_lca = self._id_to_index[lca_id]
        
        return int(self.depths[idx1] + self.depths[idx2] - 2 * self.depths[idx_lca])

    def annotate_table(self, tax_ids: Union[List[int], np.ndarray]) -> pl.DataFrame:
        """
        Massively annotates a list of TaxIDs with scientific_names and canonical ranks.
        Extremely efficient for large tables (e.g. 200k+ rows) using Polars and vectorized lookups.
        """
        logger.info(f"Annotating {len(tax_ids)} taxa...")
        
        # Build the set of canonical ranks to use for this taxonomy
        canonical_columns = [self.top_rank] + [r for r in CANONICAL_RANKS if r not in ['superkingdom', 'domain']]
        
        # Convert input to numpy array for efficient processing
        tax_ids_arr = np.array(tax_ids, dtype=np.int32)
        
        # Get internal indices
        indices = np.array([self._id_to_index.get(tid, -1) for tid in tax_ids_arr], dtype=np.int32)
        valid_mask = indices != -1
        
        # Prepare the base dictionary for Polars
        df_dict = {"tax_id": tax_ids_arr}
        
        # Vectorized lookup for each canonical rank
        for rank in canonical_columns:
            # Map indices to ancestor TaxIDs using pre-calculated maps
            ancestor_ids = np.full(len(tax_ids_arr), -1, dtype=np.int32)
            ancestor_ids[valid_mask] = self.canonical_maps[rank][indices[valid_mask]]
            
            # Map TaxIDs to Names using a dictionary lookup
            # (In Polars we can use map_dict for high performance)
            df_dict[rank] = [self.names.get(int(tid)) if tid != -1 else None for tid in ancestor_ids]

        # Add scientific_name and rank for the tax_id itself
        df_dict["scientific_name"] = [self.names.get(int(tid), "Unknown") if tid != -1 else "Unknown" for tid in tax_ids_arr]
        df_dict["rank"] = [self.rank_names[self.ranks[idx]] if idx != -1 else "unclassified" for idx in indices]
        
        df = pl.DataFrame(df_dict)
        
        # Define final column order: tax_id, then canonical ranks, then scientific_name, then rank
        final_order = ['tax_id'] + canonical_columns + ['scientific_name', 'rank']
        return df.select(final_order)

    def _ensure_up_table(self) -> None:
        """Lazy initialization of binary lifting table."""
        if self._up_table is not None:
            return
            
        logger.info("Initializing binary lifting table (Binary Lifting)...")
        num_nodes = len(self._index_to_id)
        max_log = int(np.ceil(np.log2(np.max(self.depths) + 1)))
        
        self._up_table = np.zeros((num_nodes, max_log), dtype=np.int32)
        
        # Base case: 2^0 ancestor is the parent
        for i in range(num_nodes):
            self._up_table[i, 0] = self.parents[i]
            
        # Iterative doubling
        for j in range(1, max_log):
            for i in range(num_nodes):
                self._up_table[i, j] = self._up_table[self._up_table[i, j-1], j-1]

    def save(self, directory: str) -> None:
        """Saves the vectorized tree to a directory for fast loading."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        logger.info(f"Saving binary cache to {directory}...")
        np.save(os.path.join(directory, "index_to_id.npy"), self._index_to_id)
        np.save(os.path.join(directory, "parents.npy"), self.parents)
        np.save(os.path.join(directory, "depths.npy"), self.depths)
        np.save(os.path.join(directory, "ranks.npy"), self.ranks)
        np.save(os.path.join(directory, "entry_times.npy"), self.entry_times)
        np.save(os.path.join(directory, "exit_times.npy"), self.exit_times)
        
        # Save canonical maps
        maps_dir = os.path.join(directory, "canonical_maps")
        if not os.path.exists(maps_dir):
            os.makedirs(maps_dir)
        for rank, arr in self.canonical_maps.items():
            np.save(os.path.join(maps_dir, f"{rank}.npy"), arr)
            
        import pickle
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                "names": self.names,
                "rank_names": self.rank_names,
                "id_to_index": self._id_to_index,
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
    def load(cls, directory: str) -> 'TaxonomyTree':
        """Loads the vectorized tree from a binary cache directory."""
        logger.info(f"Loading binary cache from {directory}...")
        tree = cls()
        tree._index_to_id = np.load(os.path.join(directory, "index_to_id.npy"))
        tree.parents = np.load(os.path.join(directory, "parents.npy"))
        tree.depths = np.load(os.path.join(directory, "depths.npy"))
        tree.ranks = np.load(os.path.join(directory, "ranks.npy"))
        tree.entry_times = np.load(os.path.join(directory, "entry_times.npy"))
        tree.exit_times = np.load(os.path.join(directory, "exit_times.npy"))
        
        # Load canonical maps
        maps_dir = os.path.join(directory, "canonical_maps")
        if os.path.exists(maps_dir):
            for filename in os.listdir(maps_dir):
                if filename.endswith(".npy"):
                    rank = filename[:-4]
                    tree.canonical_maps[rank] = np.load(os.path.join(maps_dir, filename))
        
        import pickle
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            meta = pickle.load(f)
            tree.names = meta["names"]
            tree.rank_names = meta["rank_names"]
            tree._id_to_index = meta["id_to_index"]
            tree.top_rank = meta.get("top_rank", "domain")
            
            # Load provenance
            prov = meta.get("provenance", {})
            tree._build_time = prov.get("build_time")
            tree._source_nodes = prov.get("source_nodes")
            tree._source_names = prov.get("source_names")
            
            logger.info("Loaded taxonomy cache:")
            logger.info(f"  Build time:    {tree._build_time}")
            logger.info(f"  Source Nodes:  {tree._source_nodes}")
            logger.info(f"  Source Names:  {tree._source_names}")
            logger.info(f"  Node count:    {prov.get('node_count', 'Unknown'):,}")
            logger.info(f"  Tree depth:    {prov.get('max_depth', 'Unknown')}")
            logger.info(f"  Top rank:      {tree.top_rank}")
            
        return tree
