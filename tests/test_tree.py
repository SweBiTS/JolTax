import unittest
import os
import sys
import numpy as np
import polars as pl

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joltax.joltree import JolTree, TaxIDNotFoundError

class TestJolTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = 'tests/data/names.dmp'
        cls.nodes = 'tests/data/nodes.dmp'
        cls.taxonomy_dir = 'tests/data/'
        # Check if files exist, if not, create them (should be copied already)
        if not os.path.exists(cls.names):
            raise FileNotFoundError(f"Missing test data: {cls.names}")
            
        cls.tree = JolTree(nodes=cls.nodes, names=cls.names)

    def test_directory_init(self):
        """Test initializing JolTree by passing a directory."""
        tree = JolTree(tax_dir=self.taxonomy_dir)
        self.assertEqual(tree.get_lineage(562), self.tree.get_lineage(562))

    def test_missing_files_error(self):
        """Test that FileNotFoundError is raised when files are missing."""
        # Test missing specific file
        with self.assertRaises(FileNotFoundError):
            JolTree(nodes='non_existent_nodes.dmp', names='non_existent_names.dmp')
        
        # Test missing files in directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                JolTree(tax_dir=tmp_dir)

    def test_lca_batch(self):
        """Test vectorized LCA batch calculation."""
        ids1 = [562, 562, 2, 1]
        ids2 = [561, 2, 562, 1]
        expected = [561, 2, 2, 1]
        
        results = self.tree.get_lca_batch(ids1, ids2)
        self.assertTrue(np.array_equal(results, expected))
        
        # Test with NumPy arrays
        results_np = self.tree.get_lca_batch(np.array(ids1), np.array(ids2))
        self.assertTrue(np.array_equal(results_np, expected))

    def test_distance_batch(self):
        """Test vectorized distance batch calculation."""
        ids1 = [562, 562]
        ids2 = [561, 2]
        expected = [1, 6]
        
        results = self.tree.get_distance_batch(ids1, ids2)
        self.assertTrue(np.array_equal(results, expected))

    def test_get_clade_at_rank(self):
        """Test get_clade_at_rank with valid and invalid ranks."""
        # 2 (Bacteria) has 562 (species)
        species_in_bacteria = self.tree.get_clade_at_rank(2, 'species')
        self.assertIn(562, species_in_bacteria)
        
        # Test invalid rank
        self.assertEqual(self.tree.get_clade_at_rank(2, 'non_existent_rank'), [])
        
        # Test node with no descendants at that rank
        self.assertEqual(self.tree.get_clade_at_rank(562, 'genus'), [])

    def test_get_indices(self):
        """Test vectorized index lookup with valid and invalid IDs."""
        ids = np.array([1, 562, 999999])
        indices = self.tree._get_indices(ids)
        self.assertEqual(len(indices), 3)
        self.assertNotEqual(indices[0], -1)
        self.assertNotEqual(indices[1], -1)
        self.assertEqual(indices[2], -1)

    def test_lca_special_cases(self):
        """Test LCA with root, same node, and missing nodes."""
        # LCA of node and itself
        self.assertEqual(self.tree.get_lca(562, 562), 562)
        # LCA with root
        self.assertEqual(self.tree.get_lca(562, 1), 1)

    def test_annotate_missing_ranks(self):
        """Test mass annotation when nodes are missing certain canonical ranks."""
        # 2 (Bacteria) is a superkingdom, so it should have None for kingdom, phylum, etc.
        df = self.tree.annotate([2])
        row = df.row(0, named=True)
        self.assertEqual(row['superkingdom'], 'Bacteria')
        self.assertIsNone(row['genus'])
        self.assertIsNone(row['species'])

    def test_search_name_edge_cases(self):
        """Test search_name with empty query and no matches."""
        # Empty query (exact)
        df = self.tree.search_name("")
        self.assertTrue(df.is_empty())
        
        # No matches (exact)
        df = self.tree.search_name("NonExistentOrganism")
        self.assertTrue(df.is_empty())
        
        # No matches (fuzzy)
        df = self.tree.search_name("XYZ123", fuzzy=True, score_cutoff=99.9)
        self.assertTrue(df.is_empty())

    def test_vectorized_cache_idempotency(self):
        """Ensure _prepare_vectorized_caches can be called multiple times safely."""
        self.tree._prepare_vectorized_caches()
        self.tree._prepare_vectorized_caches()
        # Should still work
        self.assertEqual(self.tree.get_name(562), 'Escherichia coli')

    def test_lineage(self):
        # 562 (E. coli) -> 561 (Escherichia) -> 543 -> 91347 -> 1236 -> 1224 -> 2 -> 1
        lineage = self.tree.get_lineage(562)
        expected = [1, 2, 1224, 1236, 91347, 543, 561, 562]
        self.assertEqual(lineage, expected)

    def test_clade(self):
        # Clade of 561 (genus) should contain 561 and 562 (species)
        clade = self.tree.get_clade(561)
        self.assertIn(561, clade)
        self.assertIn(562, clade)
        self.assertEqual(len(clade), 2)

    def test_lca(self):
        # LCA of 562 and 561 is 561
        lca = self.tree.get_lca(562, 561)
        self.assertEqual(lca, 561)
        
        # LCA of 562 and 2 (Bacteria) is 2
        lca = self.tree.get_lca(562, 2)
        self.assertEqual(lca, 2)

    def test_distance(self):
        # 562 to 561 is 1 step
        self.assertEqual(self.tree.get_distance(562, 561), 1)
        # 562 to 2 is 6 steps
        self.assertEqual(self.tree.get_distance(562, 2), 6)

    def test_get_name_and_rank(self):
        self.assertEqual(self.tree.get_name(562), 'Escherichia coli')
        self.assertEqual(self.tree.get_rank(562), 'species')
        self.assertEqual(self.tree.get_name(2), 'Bacteria')
        self.assertEqual(self.tree.get_rank(2), 'superkingdom')

    def test_annotate(self):
        tax_ids = [562, 561, 2]
        df = self.tree.annotate(tax_ids)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('species', df.columns)
        self.assertIn('genus', df.columns)
        
        # Check first row (562)
        row0 = df.row(0, named=True)
        self.assertEqual(row0['species'], 'Escherichia coli')
        self.assertEqual(row0['genus'], 'Escherichia')
        self.assertEqual(row0['scientific_name'], 'Escherichia coli')

    def test_annotate_single_id(self):
        """Test that annotate handles a single integer correctly."""
        df = self.tree.annotate(562)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.row(0, named=True)['scientific_name'], 'Escherichia coli')

    def test_name_search(self):
        # Search by scientific name
        df = self.tree.search_name('Escherichia coli')
        self.assertIn(562, df['tax_id'].to_list())
        
        # Search by common name
        df = self.tree.search_name('all')
        self.assertIn(1, df['tax_id'].to_list())

    def test_fuzzy_search(self):
        # Typo: "Escherchia"
        df = self.tree.search_name('Escherchia', fuzzy=True)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertTrue(len(df) > 0)
        # Top result should be Escherichia or Escherichia coli
        top_name = df.row(0, named=True)['matched_name']
        self.assertIn('Escherichia', top_name)

    def test_save_load(self):
        import shutil
        cache_dir = 'tests/cache_test'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            
        self.tree.save(cache_dir)
        new_tree = JolTree.load(cache_dir)
        
        self.assertEqual(new_tree.get_lineage(562), self.tree.get_lineage(562))
        # Check name index loaded
        df = new_tree.search_name('Escherichia coli')
        self.assertIn(562, df['tax_id'].to_list())
        
        shutil.rmtree(cache_dir)

    def test_version_validation(self):
        import shutil
        import pickle
        cache_dir = 'tests/version_test'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            
        self.tree.save(cache_dir)
        
        # Manually corrupt metadata with old version
        meta_path = os.path.join(cache_dir, "metadata.pkl")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        meta["provenance"]["package_version"] = "0.0.1" # Older than 0.1.0
        
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
            
        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            JolTree.load(cache_dir)
        
        self.assertIn("Incompatible taxonomy cache", str(cm.exception))
        shutil.rmtree(cache_dir)

class TestErrorHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = 'tests/data/names.dmp'
        cls.nodes = 'tests/data/nodes.dmp'
        cls.tree = JolTree(nodes=cls.nodes, names=cls.names)

    def test_invalid_taxid_raises_error(self):
        """Verify that invalid TaxIDs raise TaxIDNotFoundError by default."""
        invalid_id = 999999
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_name(invalid_id)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_rank(invalid_id)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_lineage(invalid_id)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_clade(invalid_id)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_lca(562, invalid_id)

    def test_valid_taxid_missing_common_name_returns_none(self):
        """Verify that valid TaxIDs with missing common names return None (no error)."""
        # ID 2 (Bacteria) has no common name in the test data
        self.assertIsNone(self.tree.get_common_name(2))
        # It should still raise error if the ID itself is missing
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_common_name(999999)

    def test_safe_mode_returns_defaults(self):
        """Verify that strict=False returns standardized safe defaults."""
        invalid_id = 999999
        self.assertIsNone(self.tree.get_name(invalid_id, strict=False))
        self.assertIsNone(self.tree.get_rank(invalid_id, strict=False))
        self.assertEqual(self.tree.get_lineage(invalid_id, strict=False), [])
        self.assertEqual(self.tree.get_clade(invalid_id, strict=False), [])
        self.assertIsNone(self.tree.get_lca(562, invalid_id, strict=False))
        self.assertIsNone(self.tree.get_distance(562, invalid_id, strict=False))

    def test_batch_strict_mode(self):
        """Verify that batch methods raise error if any ID is missing in strict mode."""
        ids1 = [562, 999999]
        ids2 = [561, 2]
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_lca_batch(ids1, ids2, strict=True)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.get_distance_batch(ids1, ids2, strict=True)
        with self.assertRaises(TaxIDNotFoundError):
            self.tree.annotate(ids1, strict=True)

    def test_batch_safe_mode(self):
        """Verify that batch methods return -1 or null in safe mode."""
        ids1 = [562, 999999]
        ids2 = [561, 2]
        
        lcas = self.tree.get_lca_batch(ids1, ids2, strict=False)
        self.assertEqual(lcas[0], 561)
        self.assertEqual(lcas[1], -1)
        
        dists = self.tree.get_distance_batch(ids1, ids2, strict=False)
        self.assertEqual(dists[0], 1)
        self.assertEqual(dists[1], -1)
        
        df = self.tree.annotate(ids1, strict=False)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.row(0, named=True)['scientific_name'], 'Escherichia coli')
        self.assertIsNone(df.row(1, named=True)['scientific_name'])

if __name__ == '__main__':
    # We skip tests if dependencies aren't installed
    try:
        import numpy
        import polars
        import rapidfuzz
        unittest.main()
    except ImportError:
        print("Skipping tests due to missing dependencies.")
