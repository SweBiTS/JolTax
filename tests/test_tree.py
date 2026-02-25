import unittest
import os
import sys
import numpy as np
import pandas as pd

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taxatree.tree import TaxonomyTree

class TestTaxonomyTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names_file = 'tests/data/names.dmp'
        cls.nodes_file = 'tests/data/nodes.dmp'
        # Check if files exist, if not, create them (should be copied already)
        if not os.path.exists(cls.names_file):
            raise FileNotFoundError(f"Missing test data: {cls.names_file}")
            
        cls.tree = TaxonomyTree(nodes_file=cls.nodes_file, names_file=cls.names_file)

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

    def test_annotate_table(self):
        tax_ids = [562, 561, 2]
        df = self.tree.annotate_table(tax_ids)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('species', df.columns)
        self.assertIn('genus', df.columns)
        self.assertEqual(df.iloc[0]['species'], 'Escherichia coli')
        self.assertEqual(df.iloc[0]['genus'], 'Escherichia')

    def test_save_load(self):
        import shutil
        cache_dir = 'tests/cache_test'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            
        self.tree.save(cache_dir)
        new_tree = TaxonomyTree.load(cache_dir)
        
        self.assertEqual(new_tree.get_lineage(562), self.tree.get_lineage(562))
        shutil.rmtree(cache_dir)

if __name__ == '__main__':
    # We skip tests if numpy/pandas aren't installed in the test environment
    try:
        import numpy
        import pandas
        unittest.main()
    except ImportError:
        print("Skipping tests due to missing numpy/pandas.")
