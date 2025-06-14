import csv
import json
import os
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import pickle

def save_dict(mol_dict, output_path):
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(mol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(input_path):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)

# 定义三字母到单字母的映射字典
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

def process_pdb_file(pdb_id, pdb_data_dir):
    protein_data = {}
    pdb_file_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_pocket.pdb')
    coords, protein_sequence = extract_protein_coords_and_seq_from_pdb(pdb_file_path)

    if not (len(coords['CA']) == len(protein_sequence) and len(coords['N']) == len(coords['C']) and
            len(coords['CA']) == len(coords['C']) and len(coords['N']) == len(coords['O'])):
        return None

    else:
        protein_dict = {
            # 'pdb_id': pdb_id,
            'seq': protein_sequence,
            'coords': coords
        }
        protein_data[pdb_id] = protein_dict
        return protein_data


splits = ["train", "valid", "test"]
# dataset = ['2016', '2020', '2021', 'LP_PDBBind', 'zinc']
# In order to avoid waiting for too much time, select one according to your requirement when running.

# ######  PDBBindv2016  ######
# for split in splits:
#     csv_file_path=f"./PDBBindv2016/last_{split}_2016.csv"
#     pdb_data_dir='/Volumes/lyz/pdb_data_v2016'
#     output_pkl_path = f'./PDBBindv2016/pocket_structures_{split}.pkl.gz'
#     invalid_pdb_ids_path = f'./PDBBindv2016/invalid_pdb_ids_{split}.json'
#     protein_dict = {}
#     invalid_pdb_ids_list = []
#
#     data = pd.read_csv(csv_file_path)
#     for index, row in data.iterrows():
#         pdb_id = row[0]
#         protein_data = process_pdb_file(pdb_id, pdb_data_dir)
#         if protein_data is None:
#             print(f"File not found for PDB ID: {pdb_id}")
#             invalid_pdb_ids_list.append(pdb_id)
#             continue
#         else:
#             for pdb_id, entry in protein_data.items():
#                 protein_dict[pdb_id] = entry
#
#     save_dict(protein_dict, output_pkl_path)
#     with open(invalid_pdb_ids_path, 'w') as json_file:
#         json.dump(invalid_pdb_ids_list, json_file, separators=(',', ':'))


# ######  PDBBindv2020  ######
# for split in splits:
#     csv_file_path=f"./PDBBindv2020/last_{split}_2020.csv"
#     pdb_data_dir='/Volumes/lyz/pdb_data_v2020'
#     output_pkl_path = f'./PDBBindv2020/pocket_structures_{split}.pkl.gz'
#     invalid_pdb_ids_path = f'./PDBBindv2020/invalid_pdb_ids_{split}.json'
#     protein_dict = {}
#     invalid_pdb_ids_list = []
#
#     data = pd.read_csv(csv_file_path)
#     for index, row in data.iterrows():
#         pdb_id = row[0]
#         protein_data = process_pdb_file(pdb_id, pdb_data_dir)
#         if protein_data is None:
#             print(f"File not found for PDB ID: {pdb_id}")
#             invalid_pdb_ids_list.append(pdb_id)
#             continue
#         else:
#             for pdb_id, entry in protein_data.items():
#                 protein_dict[pdb_id] = entry
#
#     save_dict(protein_dict, output_pkl_path)
#     with open(invalid_pdb_ids_path, 'w') as json_file:
#         json.dump(invalid_pdb_ids_list, json_file, separators=(',', ':'))


# ######  PDBBindv2021_time  ######
# for split in splits:
#     csv_file_path=f"./PDBBindv2021/PDBBindv2021_time/{split}_2021.csv"
#     pdb_data_dir='/Volumes/lyz12/pdb_data_v2021'
#     output_pkl_path = f'./PDBBindv2021/PDBBindv2021_time/pocket_structures_{split}.pkl.gz'
#     invalid_pdb_ids_path = f'./PDBBindv2021/PDBBindv2021_time/invalid_pdb_ids_{split}.json'
#     protein_dict = {}
#     invalid_pdb_ids_list = []
#
#     data = pd.read_csv(csv_file_path)
#     for index, row in data.iterrows():
#         pdb_id = row[0]
#         protein_data = process_pdb_file(pdb_id, pdb_data_dir)
#         if protein_data is None:
#             print(f"File not found for PDB ID: {pdb_id}")
#             invalid_pdb_ids_list.append(pdb_id)
#             continue
#         else:
#             for pdb_id, entry in protein_data.items():
#                 protein_dict[pdb_id] = entry
#
#     save_dict(protein_dict, output_pkl_path)
#     with open(invalid_pdb_ids_path, 'w') as json_file:
#         json.dump(invalid_pdb_ids_list, json_file, separators=(',', ':'))


######  LP_PDBBind  ######
for split in splits:
    csv_file_path=f"./lp_pdbbind/LP_PDBBind_{split}.csv"
    pdb_data_dir='/Volumes/lyz12/pdb_data_v2020'
    output_pkl_path = f'./lp_pdbbind/pocket_structures_{split}.pkl.gz'
    invalid_pdb_ids_path = f'./lp_pdbbind/invalid_pdb_ids_{split}.json'
    protein_dict = {}
    invalid_pdb_ids_list = []

    data = pd.read_csv(csv_file_path)
    for index, row in data.iterrows():
        pdb_id = row[1] # 0 or 1, please check it!
        protein_data = process_pdb_file(pdb_id, pdb_data_dir)
        if protein_data is None:
            print(f"File not found for PDB ID: {pdb_id}")
            invalid_pdb_ids_list.append(pdb_id)
            continue
        else:
            for pdb_id, entry in protein_data.items():
                protein_dict[pdb_id] = entry

    save_dict(protein_dict, output_pkl_path)
    with open(invalid_pdb_ids_path, 'w') as json_file:
        json.dump(invalid_pdb_ids_list, json_file, separators=(',', ':'))


# ######  ZINC  ######
# pdb_data_dir='./zinc/case_study_8pdf/alphafold_DoGsite3_pocket.pdb'
# output_pkl_path = f'./zinc/pocket_structures.pkl.gz'
# coords, protein_sequence = extract_protein_coords_and_seq_from_pdb(pdb_data_dir)
# assert (len(coords['CA']) == len(protein_sequence) and len(coords['N']) == len(coords['C']) and
#         len(coords['CA']) == len(coords['C']) and len(coords['N']) == len(coords['O']))
# protein_dict = {'8pdf': {
#     'seq': protein_sequence,
#     'coords': coords
# }}
# save_dict(protein_dict, output_pkl_path)

