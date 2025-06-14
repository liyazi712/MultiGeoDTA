import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import pandas as pd
import json
import os
import gzip
import pickle

def save_dict(mol_dict, output_path):
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(mol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(input_path):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)

def process_sdf_file(csv_file_path, sdf_data_dir):
    mol_dict = {}
    data = pd.read_csv(csv_file_path)
    i=1
    for index, row in data.iterrows():
        if i % 100 == 0:
            print(i)
        pdb_id = row[0] # 0 or 1, please check it!
        mol_file_path = os.path.join(sdf_data_dir, f'{pdb_id}/{pdb_id}_ligand.sdf')
        if not os.path.exists(mol_file_path):
            print(f"File not found: {mol_file_path}")
            continue

        mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)
        # 如果分子没有 3D 构象，则生成一个
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        coords = conf.GetPositions().tolist()
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        assert len(atom_types) == len(
            coords), f"Mismatch in {pdb_id}: {len(atom_types)} atoms vs {len(coords)} coordinates"
        mol_dict[pdb_id] = {
            'mol': mol,
            'coords': coords,
            'atom_types': atom_types
        }

    return mol_dict

def process_zinc_sdf_file(csv_file_path, sdf_data_dir):
    mol_dict = {}
    data = pd.read_csv(csv_file_path)
    total_rows = len(data)
    processed_rows = 0
    i=1
    for index, row in data.iterrows():
        i+=1
        if i % 100 == 0:
            print(i)
        zinc_id = row[0]
        mol_file_path = os.path.join(sdf_data_dir, f'{zinc_id}.sdf')
        if not os.path.exists(mol_file_path):
            print(f"File not found: {mol_file_path}")
            continue

        mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)
        # 如果分子没有 3D 构象，则生成一个
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        coords = conf.GetPositions().tolist()
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        assert len(atom_types) == len(
            coords), f"Mismatch in {zinc_id}: {len(atom_types)} atoms vs {len(coords)} coordinates"
        mol_dict[zinc_id] = {
            'mol': mol,
            'coords': coords,
            'atom_types': atom_types
        }
        processed_rows += 1

    print(f"CSV 文件总行数: {total_rows}")
    print(f"成功处理的分子数: {processed_rows}")
    print(f"mol_dict 中的分子数: {len(mol_dict)}")

    return mol_dict


splits = ["train", "valid", "test"]
# dataset = ['2016', '2020', '2021', 'LP_PDBBind', 'zinc']
# In order to avoid waiting for too much time, select one according to your requirement when running.

# ######  PDBBindv2016  ######
# for split in splits:
#     csv_file_path = f'./PDBBindv2016/last_{split}_2016.csv'
#     sdf_data_dir = '/Volumes/lyz/pdb_data_v2016'
#     mol_data = process_sdf_file(csv_file_path, sdf_data_dir)
#     save_dict(mol_data, f'./PDBBindv2016/mol_structures_{split}.pkl.gz')


# ######  PDBBindv2020  ######
# for split in splits:
#     csv_file_path = f'./PDBBindv2020/last_{split}_2020.csv'
#     sdf_data_dir = '/Volumes/lyz/pdb_data_v2020'
#     mol_data = process_sdf_file(csv_file_path, sdf_data_dir)
#     save_dict(mol_data, f'./PDBBindv2020/mol_structures_{split}.pkl.gz')


# ######  PDBBindv2021_time  ######
# for split in splits:
#     csv_file_path = f'./PDBBindv2021/PDBBindv2021_time/{split}_2021.csv'
#     sdf_data_dir = '/Volumes/lyz12/pdb_data_v2021'
#     mol_data = process_sdf_file(csv_file_path, sdf_data_dir)
#     save_dict(mol_data, f'./PDBBindv2021_time/mol_structures_{split}.pkl.gz')


# ######  PDBBindv2021_similarity  ######
#
# # Because the huge file will be produced, please replace <settings> and <threshold> by yourself as following:
# settings = ['new_new', 'new_compound', 'new_protein']
# thresholds = [0.3, 0.4, 0.5, 0.6]
# for setting in settings:
#     for threshold in thresholds:
#         for split in splits:
#             csv_file_path = f'./PDBBindv2021/PDBBindv2021_similarity/{setting}/{split}_{threshold}.csv'
#             sdf_data_dir = '/Volumes/lyz/pdb_data_v2021_similarity'
#             mol_data = process_sdf_file(csv_file_path, sdf_data_dir)
#             save_dict(mol_data, f'./PDBBindv2021/PDBBindv2021_similarity/{setting}/mol_structures_{split}_{threshold}.pkl.gz')


# ######  LP_PDBBind  ######
# for split in splits:
#     csv_file_path = f'./LP_PDBBind/LP_PDBBind_{split}.csv'
#     sdf_data_dir = '/Volumes/lyz12/pdb_data_v2020'
#     mol_data = process_sdf_file(csv_file_path, sdf_data_dir)
#     save_dict(mol_data, f'./lp_pdbbind/mol_structures_{split}.pkl.gz')


######  zinc  ######
# This is a test file for virtual screen, so there is no training file and valid file, and the .csv file hasn't labels.
csv_file_path = f'./zinc/code/zinc_SMILES.csv'
sdf_data_dir = './zinc/code/zinc_sdf'
mol_data = process_zinc_sdf_file(csv_file_path, sdf_data_dir)
output_pkl_path = f'./zinc/mol_structures.pkl.gz'
save_dict(mol_data, output_pkl_path)

# File not found: ./zinc/code/zinc_sdf/ZINC000123423204.sdf




