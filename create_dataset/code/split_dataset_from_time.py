import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv('pdbbind_data_last.csv')

# 筛选出2020年的数据作为测试集
test_2020 = df[df['Release_year'] == 2020].copy()

# 筛选出2020年之前的数据
remaining_data = df[df['Release_year'] != 2020].copy()

# 分割训练集和验证集
train_set, valid_set = train_test_split(remaining_data, test_size=0.1, random_state=42)

# 保存文件
train_set.to_csv('./PDBBindv2021_time/train_2021.csv', index=False)
valid_set.to_csv('./PDBBindv2021_time/valid_2021.csv', index=False)
test_2020.to_csv('./PDBBindv2021_time/test_2021.csv', index=False)

# 计算并打印每个数据集的行数
print("训练集行数: ", train_set.shape[0])
print("验证集行数: ", valid_set.shape[0])
print("测试集行数: ", test_2020.shape[0])

print("文件已成功保存！")