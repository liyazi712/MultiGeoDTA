import os

# 分子编号列表
molecule_ids = [
    "ZINC000749087800",
    "ZINC000019214839",
    "ZINC000518437019",
    "ZINC000012156381",
    "ZINC000306142100",
    "ZINC000306137110",
    "ZINC000019289571",
    "ZINC000019553307",
    "ZINC000019498244",
    "ZINC000012181259"
]

# 定义输入和输出路径
ligand_directory = "./"  # 配体文件所在的目录
# receptor_file = "alphafold_protein.pdbqt"  # 受体文件
receptor_file = "5xra_new.pdbqt"  # 受体文件
output_directory = "./dock_results"  # 输出文件保存的目录
config_template = """
ligand = {ligand}
receptor = {receptor}
out = ./dock_results/ligand_receptor_{ligand_id}.pdbqt
center_x = 89
center_y = 130
center_z = 115
size_x = 20
size_y = 20
size_z = 20
exhaustiveness = 8
seed = 1234
"""

os.makedirs(output_directory, exist_ok=True)
for mol_id in molecule_ids:
    ligand_file = os.path.join(ligand_directory, f"{mol_id}.pdbqt")
    config_file = os.path.join(output_directory, f"config_{mol_id}.txt")
    if not os.path.exists(ligand_file):
        print(f"配体文件 {ligand_file} 不存在，跳过该分子。")
        continue
    with open(config_file, 'w') as f:
        f.write(config_template.format(
            ligand=ligand_file,
            receptor=receptor_file,
            ligand_id=mol_id
        ))
    command = f"../autodock_vina_1_1_2_linux_x86/bin/vina --config {config_file}"
    os.system(command)
    print(f"已完成对分子 {mol_id} 的对接")

print("所有分子的对接已完成。")