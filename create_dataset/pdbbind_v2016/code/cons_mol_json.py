import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import csv
import json
import os

def process_sdf_file(tsv_file_path, sdf_data_dir):
    mol_dict = {}
    with open(tsv_file_path, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # 跳过标题行
        for row in reader:
            pdb_id = row[0].split('_')[1][:4]
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


splits = ["train", "valid", "test"]
for split in splits:
    tsv_file_path = f'./{split}.csv'
    sdf_data_dir = '/Volumes/lyz/pdb_data_v2016'
    mol_data = process_sdf_file(tsv_file_path, sdf_data_dir)

    output_json_path = f'./mol_structures_{split}.json'
    with open(output_json_path, 'w') as json_file:
        json.dump(mol_data, json_file, separators=(',', ':'))




