# # Batch acquisition of protein sequence, pocket sequence, and absolute position information of pockets
import pandas as pd
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

pocket_file_path = '../case_study_CB1R/alphafold_DoGsite3_pocket.pdb'
protein_file_path = '../case_study_CB1R/alphafold_protein.pdb'

protein_seq, pocket_seq, position = get_poc_seq(pocket_file_path, protein_file_path)
print(pocket_seq)
print(position)
print(protein_seq)
print(len(protein_seq), len(pocket_seq))

# 读取 CSV 文件
file_path = "./filtered_zinc_SMILES.csv"
df = pd.read_csv(file_path)

# 将 position 列表转换为字符串
position_str = ', '.join(map(str, position))  # 将列表元素转换为字符串并用逗号和空格分隔
position = f'[{position_str}]'  # 添加方括号

# 替换第三列（假设列索引为 2，因为 Python 是从 0 开始计数的）
df.insert(2, 'protein', protein_seq)
df.insert(3, 'pocket', pocket_seq)
df.insert(4, 'position', position)

# 保存到新的 CSV 文件
output_file_path = "../processed_zinc_CB1R.csv"
df.to_csv(output_file_path, index=False)

print(f"文件已成功更新,结果保存在 {output_file_path}")