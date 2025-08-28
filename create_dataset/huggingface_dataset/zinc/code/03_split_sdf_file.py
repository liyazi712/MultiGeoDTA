import os

def split_sdf(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".sdf"):
            input_path = os.path.join(input_folder, filename)

            with open(input_path, 'r') as f:
                current_molecule = []
                zinc_id = None

                for line in f:
                    current_molecule.append(line)

                    # 检测分子开始
                    if line.startswith("ZINC"):
                        zinc_id = line.strip().split()[0]  # 提取ZINC ID

                    # 检测分子结束
                    if line.strip() == "$$$$":
                        # 写入文件
                        output_path = os.path.join(output_folder, f"{zinc_id}.sdf")
                        with open(output_path, 'w') as out_file:
                            out_file.writelines(current_molecule)

                        # 重置临时存储
                        current_molecule = []
                        zinc_id = None
# 使用示例
split_sdf("./structure", "./zinc_sdf")