import os

# 分子编号列表
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
for mol_id in molecule_ids:
    command = f"cat ./5xra_new.pdb ./dock_results/ligand_receptor_{mol_id}.pdbqt > ./dock_results/complex_{mol_id}.pdb"
    os.system(command)
    print(f"已完成对分子 {mol_id} 的格式转换")

