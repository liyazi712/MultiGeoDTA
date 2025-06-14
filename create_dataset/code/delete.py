# import pandas as pd
# #
# # # 要删除的PDB ID列表
# # # pdb_ids_to_remove = [
# # #     "id_1u8t", "id_6bhe", "id_3ufa", "id_2az5", "id_2m0o", "id_4jft", "id_2qtr", "id_2p0x", "id_1o9k", "id_4os2",
# # #     "id_5f4r", "id_5t1s", "id_3uli", "id_2a0c", "id_5hdz", "id_4zy5", "id_6h4z", "id_5a3n", "id_4jik", "id_4twt",
# # #     "id_3ot3", "id_3pa4", "id_6eds", "id_5fbo"
# # # ]
# #
# # pdb_ids_to_remove = [
# #     "id_5twh"
# # ]
# #
#
# # pdb_ids_to_remove = ['4kc4', '3mmr', '3vjt', '1lvk', '3vjs', '6yoi', '6yo7']
#
# pdb_ids_to_remove = [
#     "6r1n",
#     "7c2v"
# ]
#
# # 读取CSV文件
# df = pd.read_csv('pdbbind_data_last.csv')
#
# # 假设PDB ID在第一列，列名是'pdb_id'
# # 找到要删除的PDB ID的索引
# indexes_to_remove = df.index[df['PDBname'].isin(pdb_ids_to_remove)].tolist()
#
# # 删除这些索引对应的行
# df = df.drop(indexes_to_remove)
#
# # 将结果保存到新的CSV文件
# df.to_csv('pdbbind_data_last.csv', index=False)


import pandas as pd
import os

# 定义目录路径

directory = './PDBBindv2021_similarity/new_protein'  # 替换为你的目录路径
column_names = ['PDBname', 'Smile', 'Sequence', 'Pocket', 'Position', 'label', 'Resolution', 'Release_year']

# 遍历目录中的所有CSV文件
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path).values.tolist()
        new_df = pd.DataFrame(data, columns=column_names)
        new_df.to_csv(file_path, index=False)

        print(f"Updated column names in '{filename}'")

