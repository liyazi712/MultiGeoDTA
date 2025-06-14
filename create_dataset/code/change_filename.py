import os

# 定义setting和split的组合
settings = ['new_new', 'new_protein', 'new_compound']
# splits = ['train', 'valid', 'test']
for setting in settings:
    # for split in splits:
    directory = f'./PDBBindv2021_similarity/{setting}'
    for filename in os.listdir(directory):
        parts = filename.split("_")
        if len(parts) >= 3:
            # 根据规则重命名文件
            new_name = f"{parts[0]}_{parts[2]}"
            # 构建完整的文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_name}'")