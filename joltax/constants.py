"""
joltax/constants.py
Taxonomic constants and rank mappings for joltax.
"""

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
