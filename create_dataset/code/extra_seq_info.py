# # Batch acquisition of protein sequence, pocket sequence, and absolute position information of pockets
import json
import os
import pandas as pd
import csv
from rdkit import Chem


def get_poc_seq(pocket_path, protein_path):
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
        'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
        'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
        'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}
    poc_seq = ''
    i = ''  # locator
    position = []
    protein_seq, order = get_pro_seq(protein_path)
    for line in open(pocket_path):
        if line[0:4] == "ATOM":
            columns = line.split()
            index1 = columns[4]
            index2 = columns[5]
            if len(columns[4]) > 1: # When the residue sequence exceeds 1000, there will be no spaces between the chain and sequence, and manual separation is required
                index1 = columns[4][0]
                index2 = columns[4][1:]
            if index2 != i:
                i = index2
                position.append(order[(index1, index2)])
                poc_seq += aa_codes.get(columns[3], 'X')
            else:
                continue
    return protein_seq, poc_seq, position

# This module is used to obtain the absolute position information of the entire protein sequence and pockets
def get_pro_seq(path):
    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    seq = ''
    order = {}
    i = 0  # locator
    for line in open(path):
        if line[0:4] == "ATOM":
            columns = line.split()
            index1 = columns[4]
            index2 = columns[5]
            if len(columns[4]) > 1: # When the residue sequence exceeds 1000, there will be no spaces between the chain and sequence, and manual separation is required
                index1 = columns[4][0]
                index2 = columns[4][1:]
            if (index1, index2) not in order:
                i = i + 1
                order[(index1, index2)] = i
                seq += aa_codes.get(columns[3], 'X')
            else:
                continue
    return seq, order

# def get_smile_seq(ligand_sdf_path, pdb_id):
#     mol = Chem.MolFromMolFile(ligand_sdf_path, sanitize=True)
#     smi = Chem.MolToSmiles(mol)
#     return smi

def get_smile_seq(ligand_sdf_path, pdb_id):
    try:
        mol = Chem.MolFromMolFile(ligand_sdf_path, sanitize=True)
        if mol is None:
            raise ValueError(f"Molecule could not be parsed from file: {ligand_sdf_path}")
        smi = Chem.MolToSmiles(mol)
        return smi
    except Exception as e:
        # 记录无法解析的PDB ID，这里使用pdb_id_list.txt作为记录文件
        with open('pdb_id_list.txt', 'a') as f:
            f.write(f"{pdb_id}\n")
        print(f"An error occurred while processing file {ligand_sdf_path}: {e}")
        return None


pdb_ids = []
pdb_protein = []
pdb_pocket = []
pdb_position = []
smiles =[]
label = []
missing_pdb_files = []
resolution = []
release_year = []
seq_data_path = './pdbbind_data.csv'
csv_file_path = "./pdbbind_origin_data.csv"
pdb_data_dir = '/Volumes/lyaz/pdbbind数据v2021/pdbbind_data'

with open(csv_file_path, newline="") as csvfile:
    reader = csv.reader(csvfile)
    # next(reader)
    i = 0
    for row in reader:
        i += 1
        print(i)
        pdb_id = row[1]
        pocket_file_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_pocket.pdb')
        protein_file_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_protein.pdb')
        ligand_sdf_path = os.path.join(pdb_data_dir, f'{pdb_id}/{pdb_id}_ligand.sdf')

        try:
            protein_seq, pocket_seq, position = get_poc_seq(pocket_file_path, protein_file_path)
            smile = get_smile_seq(ligand_sdf_path, pdb_id)
            pdb_ids.append(pdb_id)
            pdb_protein.append(protein_seq)
            pdb_pocket.append(pocket_seq)
            pdb_position.append(position)
            smiles.append(smile)
            label.append(row[4])
            resolution.append(row[2])
            release_year.append(row[3])
        except FileNotFoundError:
            missing_pdb_files.append(pdb_id)
            continue  # 跳过当前迭代，处理下一个PDB ID

    data = {"PDBname": pdb_ids, "Smile": smiles, "Sequence": pdb_protein, "Pocket": pdb_pocket,
            "Position": pdb_position, "label": label, 'Resolution': resolution, 'Release_year': release_year,}
    frame = pd.DataFrame(data)
    frame.to_csv(seq_data_path)

# 将缺失的PDB ID保存为JSON文件
with open(f'./missing_pdb_files_0.3.json', 'w') as json_file:
    json.dump(missing_pdb_files, json_file, indent=4)




