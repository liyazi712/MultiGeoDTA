"""
Constants and Vocabularies for MultiGeoDTA

Defines mappings and vocabularies for:
- Amino acid encoding
- Atom types
- SMILES characters
"""

# Standard amino acid three-letter to one-letter mapping
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y',  # 20 standard amino acids
    'SEC': 'U',  # Selenocysteine
    'PYL': 'O',  # Pyrrolysine
    'HYP': 'B',  # Hydroxyproline
    'SEP': 'Z',  # Phosphoserine
    'TPO': 'J'   # Phosphotyrosine
}

# Amino acid letter to number mapping (1-indexed, 0 reserved for padding)
LETTER_TO_NUM = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
    'W': 19, 'Y': 20, 'U': 21, 'O': 22, 'B': 23, 'Z': 24, 'J': 25, 'X': 26
}

# Reverse mapping: number to letter
NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}

# Atom vocabulary for molecular graphs
ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'unk'
]

# Protein character encoding with MASK token
PROTEIN_CHAR = {
    '<MASK>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
    'W': 19, 'Y': 20, 'U': 21, 'O': 22, 'B': 23, 'Z': 24, 'J': 25, 'X': 26
}

# SMILES character encoding
SMILES_CHAR = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
    ":": 0, "~": 65
}

# Electronegativity values for common atoms
ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'S': 2.58, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66
}

# Covalent radius values (in Angstroms)
COVALENT_RADIUS = {
    'H': 0.31, 'C': 0.77, 'N': 0.70, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'P': 1.07, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}

# Van der Waals radius values (in Angstroms)
VDW_RADIUS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'S': 1.80, 'P': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
}

# Valence electrons for common atoms
VALENCE_ELECTRONS = {
    'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'S': 6, 'P': 5,
    'Cl': 7, 'Br': 7, 'I': 7
}
