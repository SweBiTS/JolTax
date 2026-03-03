import unittest
import os
import sys
import numpy as np
import polars as pl

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joltax.joltree import JolTree

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
        # Test unknown
        self.assertEqual(self.tree.get_name(999999), 'Unknown_999999')
        self.assertEqual(self.tree.get_rank(999999), 'unknown')

    def test_annotate_table(self):
        tax_ids = [562, 561, 2]
        df = self.tree.annotate_table(tax_ids)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('species', df.columns)
        self.assertIn('genus', df.columns)
        
        # Check first row (562)
        row0 = df.row(0, named=True)
        self.assertEqual(row0['species'], 'Escherichia coli')
        self.assertEqual(row0['genus'], 'Escherichia')
        self.assertEqual(row0['scientific_name'], 'Escherichia coli')

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

if __name__ == '__main__':
    # We skip tests if dependencies aren't installed
    try:
        import numpy
        import polars
        import rapidfuzz
        unittest.main()
    except ImportError:
        print("Skipping tests due to missing dependencies.")
