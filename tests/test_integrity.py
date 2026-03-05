import os
import sys
import pytest
import tempfile
import shutil

# Add the project root to sys.path to ensure joltax is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joltax.joltree import JolTree
from joltax.exceptions import TaxonomyIntegrityError

@pytest.fixture
def taxonomy_files():
    test_dir = tempfile.mkdtemp()
    names_file = os.path.join(test_dir, "names.dmp")
    nodes_file = os.path.join(test_dir, "nodes.dmp")
    
    # Create a basic valid names file
    with open(names_file, "w") as f:
        f.write("1\t|\tall\t|\t\t|\tscientific name\t|\n")
        f.write("2\t|\tBacteria\t|\t\t|\tscientific name\t|\n")
        f.write("3\t|\tFirmicutes\t|\t\t|\tscientific name\t|\n")
        f.write("4\t|\tClostridia\t|\t\t|\tscientific name\t|\n")
        
    yield nodes_file, names_file
    
    shutil.rmtree(test_dir)

def write_nodes(nodes_file, nodes_data):
    with open(nodes_file, "w") as f:
        for node in nodes_data:
            f.write(f"{node[0]}\t|\t{node[1]}\t|\t{node[2]}\t|\n")

def test_missing_root(taxonomy_files):
    """Verify that missing TaxID 1 raises TaxonomyIntegrityError."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (2, 2, "superkingdom")
    ])
    with pytest.raises(TaxonomyIntegrityError, match="missing from the taxonomy nodes"):
        JolTree(nodes=nodes_file, names=names_file)

def test_orphan_node(taxonomy_files):
    """Verify that a node with a non-existent parent raises TaxonomyIntegrityError."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (1, 1, "no rank"),
        (2, 999, "species") # 999 doesn't exist
    ])
    with pytest.raises(TaxonomyIntegrityError, match="missing from the taxonomy"):
        JolTree(nodes=nodes_file, names=names_file)

def test_self_parenting_loop(taxonomy_files):
    """Verify that self-parenting (not TaxID 1) raises TaxonomyIntegrityError."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (1, 1, "no rank"),
        (2, 2, "species") # Self-loop
    ])
    with pytest.raises(TaxonomyIntegrityError, match="Self-parenting loop detected"):
        JolTree(nodes=nodes_file, names=names_file)

def test_simple_cycle(taxonomy_files):
    """Verify that a cycle (A -> B -> A) raises TaxonomyIntegrityError."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (1, 1, "no rank"),
        (2, 3, "genus"),
        (3, 2, "species") # 2 -> 3 -> 2
    ])
    with pytest.raises(TaxonomyIntegrityError, match="Cycle detected"):
        JolTree(nodes=nodes_file, names=names_file)

def test_multiple_canonical_ranks(taxonomy_files):
    """Verify that multiple nodes of the same canonical rank in a lineage raises TaxonomyIntegrityError."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (1, 1, "no rank"),
        (2, 1, "species"),
        (3, 2, "species") # Both 2 and 3 are species
    ])
    with pytest.raises(TaxonomyIntegrityError, match="Multiple nodes of canonical rank 'species'"):
        JolTree(nodes=nodes_file, names=names_file)

def test_valid_tree(taxonomy_files):
    """Verify that a valid tree builds without errors."""
    nodes_file, names_file = taxonomy_files
    write_nodes(nodes_file, [
        (1, 1, "no rank"),
        (2, 1, "superkingdom"),
        (3, 2, "phylum"),
        (4, 3, "class")
    ])
    tree = JolTree(nodes=nodes_file, names=names_file)
    assert tree.get_name(4) == "Clostridia"
