import torch
import esm
from Bio import SeqIO

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
model.set_chunk_size(128)

fasta_file = 'protein_sequences_2016.fasta'  # 替换为你的FASTA文件路径
sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))

# 用于记录因CUDA内存不足而跳过的序列的PDB ID
failed_pdb_ids = []
for seq_id, seq_record in sequences.items():
    print(f"Processing sequence {seq_id}...")
    sequence = str(seq_record.seq)  # 获取蛋白质序列
    try:
        with torch.no_grad():
            output = model.infer_pdb(sequence)

        pdb_file = f'./esm_stru/{seq_id}.pdb'
        with open(pdb_file, "w") as f:
            f.write(output)
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory for sequence {seq_id}. Skipping...")
        failed_pdb_ids.append(seq_id)  # 将PDB ID添加到失败列表中
    except Exception as e:
        print(f"An error occurred for sequence {seq_id}: {e}")

print("All sequences have been processed.")

# 将因CUDA内存不足而跳过的PDB ID保存到txt文件中
with open("failed_pdb_ids.txt", "w") as f:
    for pdb_id in failed_pdb_ids:
        f.write(f"{pdb_id}\n")