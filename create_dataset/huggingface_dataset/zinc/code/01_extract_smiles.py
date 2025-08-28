import os
import csv

input_folder = "./smile"
output_file = "./zinc_SMILES.csv"
all_data = []

for filename in os.listdir(input_folder):
    if filename.endswith(".smi"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[0:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles = parts[0]
                    zinc_id = parts[1]
                    all_data.append([zinc_id, smiles])

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['zinc_id', 'smiles'])
    writer.writerows(all_data)

print(f"数据已成功整合到 {output_file}")