import os
import sys
import pytest
import numpy as np
import polars as pl
import shutil
import pickle
import tempfile

# Add the project root to sys.path to ensure joltax is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joltax.joltree import JolTree
from joltax.exceptions import TaxIDNotFoundError

@pytest.fixture(scope="module")
def taxonomy_data():
    return {
        "names": 'tests/data/names.dmp',
        "nodes": 'tests/data/nodes.dmp',
        "dir": 'tests/data/'
    }

@pytest.fixture(scope="module")
def tree(taxonomy_data):
    if not os.path.exists(taxonomy_data["names"]):
        pytest.fail(f"Missing test data: {taxonomy_data['names']}")
    return JolTree(nodes=taxonomy_data["nodes"], names=taxonomy_data["names"])

def test_directory_init(taxonomy_data, tree):
    """Test initializing JolTree by passing a directory."""
    new_tree = JolTree(tax_dir=taxonomy_data["dir"])
    assert new_tree.get_lineage(562) == tree.get_lineage(562)

def test_missing_files_error():
    """Test that FileNotFoundError is raised when files are missing."""
    with pytest.raises(FileNotFoundError):
        JolTree(nodes='non_existent_nodes.dmp', names='non_existent_names.dmp')
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError):
            JolTree(tax_dir=tmp_dir)

def test_lca_batch(tree):
    """Test vectorized LCA batch calculation."""
    ids1 = [562, 562, 2, 1]
    ids2 = [561, 2, 562, 1]
    expected = [561, 2, 2, 1]
    
    results = tree.get_lca_batch(ids1, ids2)
    assert np.array_equal(results, expected)
    
    # Test with NumPy arrays
    results_np = tree.get_lca_batch(np.array(ids1), np.array(ids2))
    assert np.array_equal(results_np, expected)

def test_distance_batch(tree):
    """Test vectorized distance batch calculation."""
    ids1 = [562, 562]
    ids2 = [561, 2]
    expected = [1, 6]
    
    results = tree.get_distance_batch(ids1, ids2)
    assert np.array_equal(results, expected)

def test_get_clade_at_rank(tree):
    """Test get_clade_at_rank with valid and invalid ranks."""
    # 2 (Bacteria) has 562 (species)
    species_in_bacteria = tree.get_clade_at_rank(2, 'species')
    assert 562 in species_in_bacteria
    
    # Test invalid rank
    assert tree.get_clade_at_rank(2, 'non_existent_rank') == []
    
    # Test node with no descendants at that rank
    assert tree.get_clade_at_rank(562, 'genus') == []

def test_get_indices(tree):
    """Test vectorized index lookup with valid and invalid IDs."""
    ids = np.array([1, 562, 999999])
    indices = tree._get_indices(ids)
    assert len(indices) == 3
    assert indices[0] != -1
    assert indices[1] != -1
    assert indices[2] == -1

def test_lca_special_cases(tree):
    """Test LCA with root, same node, and missing nodes."""
    # LCA of node and itself
    assert tree.get_lca(562, 562) == 562
    # LCA with root
    assert tree.get_lca(562, 1) == 1

def test_annotate_missing_ranks(tree):
    """Test mass annotation when nodes are missing certain canonical ranks."""
    # 2 (Bacteria) is a superkingdom, so it should have None for kingdom, phylum, etc.
    df = tree.annotate([2])
    row = df.row(0, named=True)
    assert row['t_superkingdom'] == 'Bacteria'
    assert row['t_genus'] is None
    assert row['t_species'] is None

def test_search_name_edge_cases(tree):
    """Test search_name with empty query and no matches."""
    # Empty query (exact)
    df = tree.search_name("")
    assert df.is_empty()
    
    # No matches (exact)
    df = tree.search_name("NonExistentOrganism")
    assert df.is_empty()
    
    # No matches (fuzzy)
    df = tree.search_name("XYZ123", fuzzy=True, score_cutoff=99.9)
    assert df.is_empty()

def test_vectorized_cache_idempotency(tree):
    """Ensure _prepare_vectorized_caches can be called multiple times safely."""
    tree._prepare_vectorized_caches()
    tree._prepare_vectorized_caches()
    # Should still work
    assert tree.get_name(562) == 'Escherichia coli'

def test_lineage(tree):
    # 562 (E. coli) -> 561 (Escherichia) -> 543 -> 91347 -> 1236 -> 1224 -> 2 -> 1
    lineage = tree.get_lineage(562)
    expected = [1, 2, 1224, 1236, 91347, 543, 561, 562]
    assert lineage == expected

def test_clade(tree):
    # Clade of 561 (genus) should contain 561 and 562 (species)
    clade = tree.get_clade(561)
    assert 561 in clade
    assert 562 in clade
    assert len(clade) == 2

def test_lca(tree):
    # LCA of 562 and 561 is 561
    assert tree.get_lca(562, 561) == 561
    
    # LCA of 562 and 2 (Bacteria) is 2
    assert tree.get_lca(562, 2) == 2

def test_distance(tree):
    # 562 to 561 is 1 step
    assert tree.get_distance(562, 561) == 1
    # 562 to 2 is 6 steps
    assert tree.get_distance(562, 2) == 6

def test_get_name_and_rank(tree):
    assert tree.get_name(562) == 'Escherichia coli'
    assert tree.get_rank(562) == 'species'
    assert tree.get_name(2) == 'Bacteria'
    assert tree.get_rank(2) == 'superkingdom'

def test_annotate(tree):
    tax_ids = [562, 561, 2]
    df = tree.annotate(tax_ids)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert 't_species' in df.columns
    assert 't_genus' in df.columns
    
    # Check first row (562)
    row0 = df.row(0, named=True)
    assert row0['t_species'] == 'Escherichia coli'
    assert row0['t_genus'] == 'Escherichia'
    assert row0['t_scientific_name'] == 'Escherichia coli'

def test_annotate_single_id(tree):
    """Test that annotate handles a single integer correctly."""
    df = tree.annotate(562)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert df.row(0, named=True)['t_scientific_name'] == 'Escherichia coli'

def test_name_search(tree):
    # Search by scientific name
    df = tree.search_name('Escherichia coli')
    assert 562 in df['tax_id'].to_list()
    
    # Search by common name
    df = tree.search_name('all')
    assert 1 in df['tax_id'].to_list()

def test_fuzzy_search(tree):
    # Typo: "Escherchia"
    df = tree.search_name('Escherchia', fuzzy=True)
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0
    # Top result should be Escherichia or Escherichia coli
    top_name = df.row(0, named=True)['matched_name']
    assert 'Escherichia' in top_name

def test_tree_helpers(tree):
    """Test the available_ranks and summary properties."""
    ranks = tree.available_ranks
    assert 'species' in ranks
    assert 'genus' in ranks
    
    summary = tree.summary
    assert summary['node_count'] == 8
    assert summary['top_rank'] == 'superkingdom'
    assert isinstance(summary['max_depth'], int)

def test_save_load(tree):
    cache_dir = os.path.join(tempfile.gettempdir(), 'cache_test_pytest')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        
    tree.save(cache_dir)
    new_tree = JolTree.load(cache_dir)
    
    assert new_tree.get_lineage(562) == tree.get_lineage(562)
    # Check name index loaded
    df = new_tree.search_name('Escherichia coli')
    assert 562 in df['tax_id'].to_list()
    
    shutil.rmtree(cache_dir)

def test_version_validation(tree):
    cache_dir = os.path.join(tempfile.gettempdir(), 'version_test_pytest')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        
    tree.save(cache_dir)
    
    # Manually corrupt metadata with old version
    meta_path = os.path.join(cache_dir, "metadata.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    meta["provenance"]["package_version"] = "0.0.1" # Older than 0.1.1
    
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
        
    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Incompatible taxonomy cache"):
        JolTree.load(cache_dir)
    
    shutil.rmtree(cache_dir)

# --- Error Handling Tests ---

def test_invalid_taxid_raises_error(tree):
    """Verify that invalid TaxIDs raise TaxIDNotFoundError by default."""
    invalid_id = 999999
    with pytest.raises(TaxIDNotFoundError):
        tree.get_name(invalid_id)
    with pytest.raises(TaxIDNotFoundError):
        tree.get_rank(invalid_id)
    with pytest.raises(TaxIDNotFoundError):
        tree.get_lineage(invalid_id)
    with pytest.raises(TaxIDNotFoundError):
        tree.get_clade(invalid_id)
    with pytest.raises(TaxIDNotFoundError):
        tree.get_lca(562, invalid_id)

def test_valid_taxid_missing_common_name_returns_none(tree):
    """Verify that valid TaxIDs with missing common names return None (no error)."""
    # ID 2 (Bacteria) has no common name in the test data
    assert tree.get_common_name(2) is None
    # It should still raise error if the ID itself is missing
    with pytest.raises(TaxIDNotFoundError):
        tree.get_common_name(999999)

def test_safe_mode_returns_defaults(tree):
    """Verify that strict=False returns standardized safe defaults."""
    invalid_id = 999999
    assert tree.get_name(invalid_id, strict=False) is None
    assert tree.get_rank(invalid_id, strict=False) is None
    assert tree.get_lineage(invalid_id, strict=False) == []
    assert tree.get_clade(invalid_id, strict=False) == []
    assert tree.get_lca(562, invalid_id, strict=False) is None
    assert tree.get_distance(562, invalid_id, strict=False) is None

def test_batch_strict_mode(tree):
    """Verify that batch methods raise error if any ID is missing in strict mode."""
    ids1 = [562, 999999]
    ids2 = [561, 2]
    with pytest.raises(TaxIDNotFoundError):
        tree.get_lca_batch(ids1, ids2, strict=True)
    with pytest.raises(TaxIDNotFoundError):
        tree.get_distance_batch(ids1, ids2, strict=True)
    with pytest.raises(TaxIDNotFoundError):
        tree.annotate(ids1, strict=True)

def test_batch_safe_mode(tree):
    """Verify that batch methods return -1 or null in safe mode."""
    ids1 = [562, 999999]
    ids2 = [561, 2]
    
    lcas = tree.get_lca_batch(ids1, ids2, strict=False)
    assert lcas[0] == 561
    assert lcas[1] == -1
    
    dists = tree.get_distance_batch(ids1, ids2, strict=False)
    assert dists[0] == 1
    assert dists[1] == -1
    
    df = tree.annotate(ids1, strict=False)
    assert len(df) == 2
    assert df.row(0, named=True)['t_scientific_name'] == 'Escherichia coli'
    assert df.row(1, named=True)['t_scientific_name'] is None

# --- Type Guard Tests ---

def test_scalar_type_guards(tree):
    """Verify that scalar methods raise TypeError for non-integer TaxIDs."""
    invalid_input = "562" # String instead of int
    with pytest.raises(TypeError):
        tree.get_name(invalid_input)
    with pytest.raises(TypeError):
        tree.get_rank(invalid_input)
    with pytest.raises(TypeError):
        tree.get_lineage(invalid_input)
    with pytest.raises(TypeError):
        tree.get_clade(invalid_input)
    with pytest.raises(TypeError):
        tree.get_clade_at_rank(invalid_input, 'species')
    with pytest.raises(TypeError):
        tree.get_lca(invalid_input, 561)
    with pytest.raises(TypeError):
        tree.get_lca(561, invalid_input)

def test_batch_type_guards(tree):
    """Verify that batch methods raise TypeError for non-iterable inputs."""
    invalid_input = 562 # int instead of list/array
    with pytest.raises(TypeError):
        tree.get_lca_batch(invalid_input, [561])
    with pytest.raises(TypeError):
        tree.get_distance_batch([561], invalid_input)

def test_annotate_type_guard(tree):
    """Verify that annotate raises TypeError for invalid types."""
    invalid_input = {"id": 562} # dict instead of int/list/array
    with pytest.raises(TypeError):
        tree.annotate(invalid_input)
