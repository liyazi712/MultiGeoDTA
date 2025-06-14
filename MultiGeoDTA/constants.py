
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y',  # 20种标准氨基酸
    'SEC': 'U',  # 硒代半胱氨酸 (Selenocysteine)
    'PYL': 'O',  # 吡咯赖氨酸 (Pyrrolysine)
    'HYP': 'B',  # 羟脯氨酸 (Hydroxyproline) 将其从原本的'X'改为'B'，可以避免与未知氨基酸的冲突
    'SEP': 'Z',  # 磷酸丝氨酸 (Phosphoserine)
    'TPO': 'J'   # 磷酸酪氨酸 (Phosphotyrosine)
}

LETTER_TO_NUM = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
    'W': 19, 'Y': 20, 'U': 21, 'O': 22, 'B': 23, 'Z': 24, 'J': 25, 'X': 26
}


NUM_TO_LETTER = {v:k for k, v in LETTER_TO_NUM.items()}

ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'unk']

# ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'unk']
