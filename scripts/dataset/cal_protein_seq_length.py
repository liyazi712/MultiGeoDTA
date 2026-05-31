import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('pdbbind_data_last.csv')

# 计算每个序列的长度
df['Sequence_Length'] = df['Sequence'].apply(len)

# 计算均值和标准差
mean_length = df['Sequence_Length'].mean()
std_length = df['Sequence_Length'].std()

# 准备画图
plt.figure(figsize=(10, 6))
plt.hist(df['Sequence_Length'], bins=50, color='blue', edgecolor='black')

# 在均值和3个标准差处画竖线
plt.axvline(x=mean_length, color='red', linestyle='--', label='Mean')
plt.axvline(x=mean_length + std_length, color='green', linestyle='--', label='Mean + 1*STD')
plt.axvline(x=mean_length + 2*std_length, color='green', linestyle='--', label='Mean + 2*STD')
plt.axvline(x=mean_length + 3*std_length, color='green', linestyle='--', label='Mean + 3*STD')

# 设置图例
plt.legend()

# 设置标题和标签
plt.title('Sequence Length Histogram with Mean and 3 Sigma')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')

# 保存图形为PDF文件
plt.savefig('sequence_length_histogram.pdf', bbox_inches='tight')

# 显示图形
plt.show()