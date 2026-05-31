# Dataset preparation scripts

Legacy script names from the original repo are preserved. Recommended names:

| Legacy file | Purpose |
|-------------|---------|
| `cons_pocket_json.py` | Build pocket structure JSON |
| `cons_mol_json.py` | Build molecule JSON |
| `split_dataset_from_time.py` | Time-based split |
| `split_dataset_from_similarity.py` | Similarity-based split |
| `cluster_based_similarity.py` | Cluster compounds/proteins |
| `preprocess_pocket_pkl.py` | Pocket structures → pkl.gz |
| `preprocess_mol_pkl.py` | Molecule structures → pkl.gz |
| `copy_sdf_files.py` | Copy SDF files |
| `cal_DDE_features_2021.py` | DDE features for PDBBind 2021 |

Run from repository root. Point `MULTIGEODTA_DATA_DIR` to your data folder.
