import csv
import json
import os
from Bio.PDB import PDBParser

# 定义三字母到单字母的映射字典
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
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
            aa_one_letter_coords = three_to_one_coords(aa_three_letter)  # 获取单字母表示
            if aa_one_letter_coords == 'X':  # 如果是非标准氨基酸
                continue
            for atom in residue:
                if atom.name.strip() in ['N', 'CA', 'C', 'O']:
                    coord = list(round(float(x), 5) for x in atom.coord)  # 转换为原生浮点数的元组
                    coords[atom.name.strip()].append(coord)
                    # coords[atom.name.strip()].append(list(atom.coord))
            coords = {k: [list(item) for item in v] for k, v in coords.items()}

    return coords, ''.join(protein_sequence)

def process_invalid_pdb_data(pdb_id, pdb_data_dir):
    pdb_file_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_pocket.pdb')
    coords, protein_sequence = extract_protein_coords_and_seq_from_pdb(pdb_file_path)
    if not (len(coords['CA']) == len(protein_sequence) and len(coords['N']) == len(coords['C']) and
            len(coords['CA']) == len(coords['C']) and len(coords['N']) == len(coords['O'])):
        return pdb_id
    return None

def process_pdb_file(pdb_id, pdb_data_dir):
    protein_data = {}
    pdb_file_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_pocket.pdb')
    coords, protein_sequence = extract_protein_coords_and_seq_from_pdb(pdb_file_path)
    assert len(coords['CA'])==len(protein_sequence) and len(coords['N'])==len(coords['C']) and len(coords['CA'])==len(coords['C']) and len(coords['N'])==len(coords['O'])
    protein_dict = {
        'pdb_id': pdb_id,
        'seq': protein_sequence,
        'coords': coords
    }
    protein_data[pdb_id] = protein_dict
    return protein_data

splits = ["train", "valid", "test"]
settings = ['new_new', 'new_compound', 'new_protein']

for setting in settings:
    for split in splits:
        csv_file_path = f'./PDBBindv2021_similarity/{setting}/{split}_0.6.csv'
        pdb_data_dir = '/Volumes/lyaz/pdbbind数据v2021/pdbbind_data'

        output_json_path = f'./PDBBindv2021_similarity/{setting}/pocket_structures_{split}_0.6.json'
        invalid_pdb_ids_path = f'./PDBBindv2021_similarity/{setting}/invalid_pdb_ids_{split}.json'
        protein_dict = {}
        invalid_pdb_ids_list = [] # 用于去除CA,C,N,O四个原子个数不一样以及文件不存在的pdb entry，主要是前者
        with open(csv_file_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            i = 0
            for row in reader:
                i += 1
                print(i)
                try:
                    pdb_id = row[0]
                except IndexError:
                    print(f"Error: Invalid PDB ID format: {row[0]}")
                    continue

                try:
                    invalid_pdb_id = process_invalid_pdb_data(pdb_id, pdb_data_dir)
                    if invalid_pdb_id:
                        invalid_pdb_ids_list.append(invalid_pdb_id) # 用于去除CA,C,N,O四个原子个数不一样的pdb entry
                        continue

                    protein_data = process_pdb_file(pdb_id, pdb_data_dir)
                    for pdb_id, entry in protein_data.items():
                        protein_dict[pdb_id] = entry

                except FileNotFoundError:
                    print(f"File not found for PDB ID: {pdb_id}")
                    invalid_pdb_ids_list.append(pdb_id)
                    continue

        with open(output_json_path, 'w') as json_file:
            json.dump(protein_dict, json_file, separators=(',', ':'))

        with open(invalid_pdb_ids_path, 'w') as json_file:
            json.dump(invalid_pdb_ids_list, json_file, separators=(',', ':'))