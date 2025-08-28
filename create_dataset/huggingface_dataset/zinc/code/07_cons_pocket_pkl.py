from Bio.PDB import PDBParser
import gzip
import pickle

def save_dict(mol_dict, output_path):
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(mol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


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

def three_to_one_coords(aa_three_letter):
    # 对于非常见的20种氨基酸，返回 'X'
    return AA_MAP.get(aa_three_letter, 'X')

def three_to_one_seq(aa_three_letter):
    # 如果是常见的20种氨基酸，返回其单字母表示；否则，返回 None
    return AA_MAP.get(aa_three_letter, None)

def extract_protein_coords_and_seq_from_pdb(pdb_file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    protein_sequence = []
    coords = {'N': [], 'CA': [], 'C': [], 'O': []}

    for residue in structure.get_residues():
        if residue.has_id('CA'):
            aa_three_letter = residue.get_resname()
            aa_one_letter_seq = three_to_one_seq(aa_three_letter)
            if aa_one_letter_seq:  # 如果不是 None，即不是非标准氨基酸
                protein_sequence.append(aa_one_letter_seq)
            aa_one_letter_coords = three_to_one_coords(aa_three_letter)
            if aa_one_letter_coords == 'X':  # 如果是非标准氨基酸
                continue
            for atom in residue:
                if atom.name.strip() in ['N', 'CA', 'C', 'O']:
                    coord = list(round(float(x), 3) for x in atom.coord)
                    coords[atom.name.strip()].append(coord)
                    # coords[atom.name.strip()].append(list(atom.coord))
            coords = {k: [list(item) for item in v] for k, v in coords.items()}

    return coords, ''.join(protein_sequence)

pdb_data_dir='../case_study_CB1R/alphafold_DoGsite3_pocket.pdb'
output_pkl_path = f'../pocket_structures.pkl.gz'
coords, protein_sequence = extract_protein_coords_and_seq_from_pdb(pdb_data_dir)
assert (len(coords['CA']) == len(protein_sequence) and len(coords['N']) == len(coords['C']) and
        len(coords['CA']) == len(coords['C']) and len(coords['N']) == len(coords['O']))
protein_dict = {'target': {
    'seq': protein_sequence,
    'coords': coords
}}
save_dict(protein_dict, output_pkl_path)