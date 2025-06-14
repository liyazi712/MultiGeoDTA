################################################################################################
This is a readme file to help create the dataset which contains our requirements.The construction
of PDBBindv2021_time and PDBBindv2021_similarity follows the same precedure.
################################################################################################

1. run extra_pdb_entry.py
input:
output: pdbbind_origin_data.csv

2. run extra_seq_info.py
input: pdbbind_origin_data.csv
output: pdbbind_data.csv, pdb_id_list.txt

3. run delete.py
input: pdbbind_data.csv, pdb_id_list.txt
output: pdbbind_data_last.csv

4. run split_dataset_from_time.py
input: pdbbind_data_last.csv
output: train_2021.csv, valid_2021.csv, test_2021.csv

5. run cons_pocket_json.py
input: train_2021.csv, valid_2021.csv, test_2021.csv
output: pocket_structures_{setting}.json, invalid_pdb_ids_{setting}.json (assert contain no pdb_id)

6. run cons_mol_json.py
input: train_2021.csv, valid_2021.csv, test_2021.csv
output: mol_structures_{setting}.json

###################################################################################################

7. run cal_DDE_features_2021.py
input: pdbbind_data_last.csv
output: protein_unique_list_2021.fasta

8. upload protein_unique_list_2021.fasta into the iLearnPlus platform to calculate protein DDE features
input: protein_unique_list_2021.fasta
output: DDE_features_2021.csv

9. run cluster_based_similarity.py
input: pdbbind_data_last.csv
output: compound_cluster_dict_{thre}, protein_cluster_dict_{thre}

10. run split_dataset_from_similarity.py
input: compound_cluster_dict_{thre}, protein_cluster_dict_{thre}
output: ./PDBBindv2021_similarity/{setting}/{split}_{fold}_{thre}.csv

11. select dataset according to the number of samples
input: ./PDBBindv2021_similarity/{setting}/{split}_{fold}_{thre}.csv
output: ./PDBBindv2021_similarity/{setting}/{split}_{thre}.csv

12. adjust file: run delete.py, change_filename.py

13. run cons_pocket_json.py
input: train_0.5.csv, valid_0.5.csv, test_0.5.csv
output: pocket_structures_{setting}_0.5.json, invalid_pdb_ids_{setting}.json (assert contain no pdb_id)

14. run cons_mol_json.py
input: train_0.5.csv, valid_0.5.csv, test_0.5.csv
output: mol_structures_{setting}_0.5.json
