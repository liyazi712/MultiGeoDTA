import csv
import os
from pathlib import Path
import shutil


def copy_sdf_files(src_dir, dest_dir, valid_pdbid_list):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for pdbid in valid_pdbid_list:
        src_pdb_id_dir = Path(src_dir) / pdbid
        # if src_pdb_id_dir.is_dir():
        sdf_filename = f"{pdbid}_ligand.sdf"
        src_sdf_file_path = src_pdb_id_dir / sdf_filename
        if src_sdf_file_path.exists():
            dest_file_path = Path(dest_dir) / sdf_filename
            shutil.copy(src_sdf_file_path, dest_file_path)
            print(f"Copied: {src_sdf_file_path} to {dest_file_path}")
        else:
            print(f"File not found: {src_sdf_file_path}")


# 读取有效PDB ID列表
# valid_pdbid_list = []
# with open('./seq_dataset/out3_last_seq_data_All.tsv') as f:
#     next(f)  # 跳过第一行
#     for line in f:
#         lines = line.strip().split('\t')
#         valid_pdbid_list.append(lines[1])

# 读取有效PDB ID列表
valid_pdbid_list = []
with open('./pdbbind_data_last.csv', mode='r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过第一行（标题行）
    for row in reader:
        valid_pdbid_list.append(row[0])  # 读取第二列

# 定义源文件夹和目标文件夹路径
src_dir = '/Volumes/lyaz/pdbbind数据v2021/pdbbind_data'
# dest_dir = './pdbbind_v2021_mol3d_sdf'
dest_dir = './pdbbind_v2021_mol3d_sdf'

# 调用函数
copy_sdf_files(src_dir, dest_dir, valid_pdbid_list)