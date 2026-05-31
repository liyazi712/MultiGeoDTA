import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import pandas as pd
import json
import os
import gzip
import pickle
from tqdm import tqdm

def save_dict(mol_dict, output_path):
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(mol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(input_path):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)

def process_sdf_file(csv_file_path, sdf_data_dir):
    mol_dict = {}
    data = pd.read_csv(csv_file_path)
    missing = 0
    for row in tqdm(data.itertuples(index=False), total=len(data), desc="PDBBind ligand graphs", unit="mol"):
        pdb_id = row[0]
        mol_file_path = os.path.join(sdf_data_dir, f'{pdb_id}/{pdb_id}_ligand.sdf')
        if not os.path.exists(mol_file_path):
            missing += 1
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

    if missing:
        tqdm.write(f"Warning: {missing} SDF files not found (skipped)")
    tqdm.write(f"Saved {len(mol_dict)} / {len(data)} ligand graphs")
    return mol_dict

def process_zinc_sdf_file(csv_file_path, sdf_data_dir):
    mol_dict = {}
    data = pd.read_csv(csv_file_path)
    total_rows = len(data)
    missing = 0
    for row in tqdm(data.itertuples(index=False), total=total_rows, desc="ZINC ligand graphs", unit="mol"):
        zinc_id = row[0]
        mol_file_path = os.path.join(sdf_data_dir, f'{zinc_id}.sdf')
        if not os.path.exists(mol_file_path):
            missing += 1
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

    if missing:
        tqdm.write(f"Warning: {missing} SDF files not found (skipped)")
    tqdm.write(f"Saved {len(mol_dict)} / {total_rows} ZINC ligand graphs")
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
if __name__ == "__main__":
    # Virtual screen: no train/valid split; CSV has no labels.
    csv_file_path = "./zinc/code/zinc_SMILES.csv"
    sdf_data_dir = "./zinc/code/zinc_sdf"
    mol_data = process_zinc_sdf_file(csv_file_path, sdf_data_dir)
    output_pkl_path = "./zinc/mol_structures.pkl.gz"
    save_dict(mol_data, output_pkl_path)




