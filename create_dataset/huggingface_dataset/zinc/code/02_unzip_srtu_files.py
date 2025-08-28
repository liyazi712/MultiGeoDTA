import gzip
import shutil
import os

# 设置文件夹路径（替换为你的路径）
folder_path = "./sdf"

for filename in os.listdir(folder_path):
    if filename.endswith(".sdf.gz"):
        # 构建完整文件路径
        gz_path = os.path.join(folder_path, filename)
        # 生成解压后的文件名（去掉.gz后缀）
        output_name = filename[:-3]  # 假设文件名类似 BAAAMO.xaa.sdf.gz
        output_path = os.path.join('./structure', output_name)

        # 执行解压
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"解压完成: {filename} → {output_name}")

print("所有文件已处理完毕！")
