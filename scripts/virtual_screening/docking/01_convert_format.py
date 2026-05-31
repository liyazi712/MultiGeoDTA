import os

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
input_directory = "./"  # 输入文件所在的目录
output_directory = "./"  # 输出文件保存的目录

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 循环处理每个分子
for mol_id in molecule_ids:
    input_file = os.path.join(input_directory, f"{mol_id}.sdf")
    output_file = os.path.join(output_directory, f"{mol_id}.pdbqt")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在，跳过该分子。")
        continue
    command = f"obabel {input_file} -O {output_file}"
    os.system(command)
    print(f"已处理 {mol_id}")