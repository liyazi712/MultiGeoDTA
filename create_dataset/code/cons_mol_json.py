import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import csv
import json
import os
import numpy as np

ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'unk']

def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]
    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)
    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values

def _build_atom_feature(mol):
    # dim: 44 + 7 + 7 + 7 + 1
    feature_alphabet = {
        # (alphabet, default value)
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }
    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                    alphabet=feature_alphabet[attr][0],
                    default=feature_alphabet[attr][1],
                    drop_first=(attr in ['GetIsAromatic']) # binary-class feature
                )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)
    return atom_feature

# def process_sdf_file(tsv_file_path, sdf_data_dir):
#     mol_dict = {}
#     with open(tsv_file_path, newline='') as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#         next(reader)  # 跳过标题行
#         for row in reader:
#             # pdb_id = row[0].split('_')[1][:4]
#             pdb_id = row[0]
#             mol_file_path = os.path.join(sdf_data_dir, f'{pdb_id}/{pdb_id}_ligand.sdf')
#             if not os.path.exists(mol_file_path):
#                 print(f"File not found: {mol_file_path}")
#                 continue
#
#             # 读取 SDF 文件并获取分子
#             mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)
#             # 如果分子没有 3D 构象，则生成一个
#             if mol.GetNumConformers() == 0:
#                 AllChem.EmbedMolecule(mol)
#             conf = mol.GetConformer()
#             coords = conf.GetPositions().tolist()
#             atoms_feature = _build_atom_feature(mol).tolist()
#             mol_dict[pdb_id] = {
#                 'atoms_feature': atoms_feature,
#                 'coords': coords
#             }
#
#     return mol_dict

def process_sdf_file(csv_file_path, sdf_data_dir):
    mol_dict = {}
    data = pd.read_csv(csv_file_path)
    for index, row in data.iterrows():
        pdb_id = row[0]
        mol_file_path = os.path.join(sdf_data_dir, f'{pdb_id}/{pdb_id}_ligand.sdf')
        if not os.path.exists(mol_file_path):
            print(f"File not found: {mol_file_path}")
            continue

        # 读取 SDF 文件并获取分子
        mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)
        # 如果分子没有 3D 构象，则生成一个
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        coords = conf.GetPositions().tolist()
        atoms_feature = _build_atom_feature(mol).tolist()
        mol_dict[pdb_id] = {
            'mol': mol,
            'atoms_feature': atoms_feature,
            'coords': coords
        }

    return mol_dict


splits = ["train", "valid", "test"]
settings = ['new_new', 'new_compound', 'new_protein']

for setting in settings:
    for split in splits:
        # 设置 TSV 文件路径和 SDF 数据目录路径
        csv_file_path = f'./PDBBindv2021_similarity/{setting}/{split}_0.6.csv'
        sdf_data_dir = '/Volumes/lyaz/pdbbind数据v2021/pdbbind_data'

        # 处理 SDF 文件并获取分子坐标字典
        mol_data = process_sdf_file(csv_file_path, sdf_data_dir)

        # 输出 JSON 文件
        output_json_path = f'./PDBBindv2021_similarity/{setting}/mol_structures_{split}_0.6.json'
        with open(output_json_path, 'w') as json_file:
            json.dump(mol_data, json_file, separators=(',', ':'))

