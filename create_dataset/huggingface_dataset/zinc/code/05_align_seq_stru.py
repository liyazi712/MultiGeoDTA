import csv
import gzip
import pickle

# 定义文件路径
pkl_file_path = "../mol_structures.pkl.gz"
csv_file_path = "zinc_SMILES.csv"
output_file_path = "filtered_zinc_SMILES.csv"

# 获取 JSON 文件中的所有 ID
def load_dict(input_path):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)

data = load_dict(pkl_file_path)
ids = set(data.keys())
print(f"pkl 文件中不同 ID 的数量: {len(ids)}")

# 筛选 CSV 文件中与 JSON 文件具有相同 ID 的项
with open(csv_file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
    csv_reader = csv.DictReader(csv_file)
    csv_writer = csv.DictWriter(output_file, fieldnames=csv_reader.fieldnames)
    csv_writer.writeheader()

    matched_ids = set()
    for row in csv_reader:
        zinc_id = row.get('zinc_id')
        if zinc_id and zinc_id in ids:
            csv_writer.writerow(row)
            matched_ids.add(zinc_id)

    print(f"CSV 文件中匹配的 ID 数量: {len(matched_ids)}")
    print(f"筛选后的 CSV 文件已保存到 {output_file_path}")

# 验证两者所包含的 ID 是否一致
print(f"pkl 文件中的 ID 数量: {len(ids)}")
print(f"筛选后的 CSV 文件中的 ID 数量: {len(matched_ids)}")
print(f"两者所包含的 ID 是否一致: {len(ids) == len(matched_ids)}")