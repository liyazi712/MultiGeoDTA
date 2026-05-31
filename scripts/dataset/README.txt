###########################################
This is a readme file to help create the dataset which contains our requirements.The construction
of PDBBindv2016 and PDBBindv2020 follows the same precedure.
###########################################

1. prepare train.csv, valid.csv, test.csv which must contains pdb_id, smile, protein_sequence, affinity value.

2. prepare structure data which can be downloaded from pdbbind database

3. run cons_pocket_json.py
note: please make sure the invalid_pdb_ids_train/valid/test.json file contains no entry,
if no, please remove the entry from the train/valid/test.csv file.

4. run cons_mol_json.py
note: please make sure the number of entry keep the same.

5. run copy_sdf_file.py to construct the pdbbind_v2016_mol3d_sdf file


