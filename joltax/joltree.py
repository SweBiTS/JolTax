#!/usr/bin/env python3
"""
joltax/joltree.py
Implementation of a high-performance, vectorized taxonomy tree.
"""

__version__ = "0.1.1"

# The minimum version of a saved taxonomy cache that is compatible with this software.
# Increment this when making breaking changes to the binary layout or metadata structure.
MINIMUM_CACHE_VERSION = "0.1.1"

import logging
import os
import datetime
from typing import Dict, List, Optional, Set, Union, Tuple
from collections import namedtuple

import numpy as np
import polars as pl
from rapidfuzz import process, fuzz, utils

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

class JolTree:
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
        scientific_names = {}
        common_names = {}
        search_data = [] # List of (name, tax_id)
        
        with open(names_file, 'r') as f:
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
            while True:
                rank_name = self.rank_names[self.ranks[curr_idx]]
                
                # Normalize superkingdom/domain based on detected top_rank
                mapped_rank = rank_name
                if rank_name in ['superkingdom', 'domain']:
                    mapped_rank = self.top_rank
                
                if mapped_rank in self.canonical_maps:
                    self.canonical_maps[mapped_rank][i] = curr_idx
                
                if curr_idx == root_idx:
                    break
                curr_idx = self.parents[curr_idx]

    def _calculate_depth(self, index: int) -> int:
        """Recursive depth calculation with memoization."""
        if index == 0: # TaxID 1 is always index 0
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
        """Returns the internal index for a TaxID, or -1 if not found."""
        idx = np.searchsorted(self._index_to_id, tax_id)
        if idx < len(self._index_to_id) and self._index_to_id[idx] == tax_id:
            return int(idx)
        return -1
    
    def _get_indices(self, tax_ids: np.ndarray) -> np.ndarray:
        """Returns internal indices for an array of TaxIDs, with -1 for missing."""
        indices = np.searchsorted(self._index_to_id, tax_ids)
        # Handle out of bounds
        mask = indices < len(self._index_to_id)
        # Check for actual equality
        valid = np.zeros(len(tax_ids), dtype=bool)
        valid[mask] = self._index_to_id[indices[mask]] == tax_ids[mask]
        return np.where(valid, indices, -1)

    def get_lineage(self, tax_id: int) -> List[int]:
        """Returns the full lineage from root to the given TaxID."""
        idx = self._get_index(tax_id)
        if idx == -1:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        lineage = []
        root_idx = 0
        
        while True:
            lineage.append(int(self._index_to_id[idx]))
            if idx == root_idx:
                break
            idx = self.parents[idx]
            
        return lineage[::-1]

    def get_name(self, tax_id: int) -> str:
        """Returns the scientific name of the given TaxID."""
        idx = self._get_index(tax_id)
        if idx != -1:
            return self._scientific_names[idx]
        return f"Unknown_{tax_id}"

    def get_common_name(self, tax_id: int) -> Optional[str]:
        """Returns the genbank common name of the given TaxID, if available."""
        idx = self._get_index(tax_id)
        if idx != -1:
            return self._common_names[idx]
        return None

    def get_rank(self, tax_id: int) -> str:
        """Returns the taxonomic rank of the given TaxID."""
        idx = self._get_index(tax_id)
        if idx == -1:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return "unknown"
        return self.rank_names[self.ranks[idx]]

    def search_name(self, query: str, fuzzy: bool = False, limit: int = 10, score_cutoff: float = 60.0) -> pl.DataFrame:
        """
        Searches for TaxIDs by name.
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

    def get_clade(self, tax_id: int) -> List[int]:
        """Returns all TaxIDs in the clade rooted at the given TaxID."""
        idx = self._get_index(tax_id)
        if idx == -1:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
            return []
        
        entry = self.entry_times[idx]
        exit = self.exit_times[idx]
        
        mask = (self.entry_times >= entry) & (self.entry_times <= exit)
        return self._index_to_id[mask].astype(int).tolist()

    def get_clade_at_rank(self, tax_id: int, rank_name: str) -> List[int]:
        """
        Returns all TaxIDs of a specific rank within the clade rooted at tax_id.
        """
        idx = self._get_index(tax_id)
        if idx == -1:
            logger.warning(f"TaxID {tax_id} not found in taxonomy tree.")
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

    def get_lca(self, tax_id_1: int, tax_id_2: int) -> int:
        """Finds the Lowest Common Ancestor using Binary Lifting."""
        idx1 = self._get_index(tax_id_1)
        idx2 = self._get_index(tax_id_2)
        
        if idx1 == -1 or idx2 == -1:
            logger.warning(f"One or both TaxIDs ({tax_id_1}, {tax_id_2}) not found.")
            return 1
            
        self._ensure_up_table()
        
        if self.depths[idx1] < self.depths[idx2]:
            idx1, idx2 = idx2, idx1
        
        diff = self.depths[idx1] - self.depths[idx2]
        for i in range(self._up_table.shape[1]):
            if (diff >> i) & 1:
                idx1 = self._up_table[idx1, i]
                
        if idx1 == idx2:
            return int(self._index_to_id[idx1])
            
        for i in reversed(range(self._up_table.shape[1])):
            if self._up_table[idx1, i] != self._up_table[idx2, i]:
                idx1 = self._up_table[idx1, i]
                idx2 = self._up_table[idx2, i]
                
        return int(self._index_to_id[self.parents[idx1]])

    def get_distance(self, tax_id_1: int, tax_id_2: int) -> int:
        """Calculates distance (number of edges) between two TaxIDs."""
        lca_id = self.get_lca(tax_id_1, tax_id_2)
        idx1 = self._get_index(tax_id_1)
        idx2 = self._get_index(tax_id_2)
        idx_lca = self._get_index(lca_id)
        return int(self.depths[idx1] + self.depths[idx2] - 2 * self.depths[idx_lca])

    def annotate_table(self, tax_ids: Union[List[int], np.ndarray]) -> pl.DataFrame:
        """
        Massively annotates a list of TaxIDs with scientific_names and canonical ranks.
        Extremely efficient for large tables (e.g. 200k+ rows) using Polars and vectorized lookups.
        """
        logger.info(f"Annotating {len(tax_ids)} taxa...")
        canonical_columns = [self.top_rank] + [r for r in CANONICAL_RANKS if r not in ['superkingdom', 'domain']]
        
        tax_ids_arr = np.array(tax_ids, dtype=np.int32)
        indices = self._get_indices(tax_ids_arr)
        valid_mask = indices != -1
        
        # dummy_idx points to the "Unknown/None" entry at the end of the lookup series
        dummy_idx = len(self._index_to_id)
        safe_indices = np.where(valid_mask, indices, dummy_idx)
        
        # Ensure caches are ready
        if self._sci_names_lookup is None:
            self._prepare_vectorized_caches()
            
        df_dict = {"tax_id": tax_ids_arr}
        
        for rank in canonical_columns:
            # canonical_maps now store internal indices
            ancestor_indices = np.full(len(tax_ids_arr), -1, dtype=np.int32)
            # Map input tax_ids to their ancestor's internal index
            ancestor_indices[valid_mask] = self.canonical_maps[rank][indices[valid_mask]]
            
            # Use dummy_idx for missing ancestors
            safe_anc_indices = np.where(ancestor_indices != -1, ancestor_indices, dummy_idx)
            
            # Vectorized gather from Polars
            df_dict[rank] = self._sci_names_lookup.gather(safe_anc_indices.astype(np.int32))

        # Scientific name for the input TaxID
        df_dict["scientific_name"] = self._sci_names_lookup.gather(safe_indices.astype(np.int32))
        
        # Rank for the input TaxID
        target_rank_indices = self._ranks_extended[safe_indices]
        df_dict["rank"] = self._rank_names_series.gather(target_rank_indices.astype(np.int32))
        
        df = pl.DataFrame(df_dict)
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
        for i in range(num_nodes):
            self._up_table[i, 0] = self.parents[i]
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
        """Loads the vectorized tree from a binary cache directory."""
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
