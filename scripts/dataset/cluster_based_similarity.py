import csv
import os
import pickle
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import substitution_matrices
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import fcluster, single
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity



def get_fps(mol_list):
    fps = []
    i = 0
    for smile in mol_list:
        i = i + 1
        print(i)
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            # 计算 Morgan 指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
            fps.append(fp)
        else:
            print(f"Warning: Unable to parse SMILES string '{smile}' into a molecule.")
    return fps


def calculate_sims(fps1,fps2,simtype='tanimoto'):
    sim_mat = np.zeros((len(fps1),len(fps2))) #,dtype=np.float32)
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)
        sim_mat[i,:] = sims
    return sim_mat


def compound_clustering(mol_list):
    print('start compound clustering...')
    ligand_uniqueness = list(set(mol_list))
    fps = get_fps(ligand_uniqueness)
    # ligand_uniqueness = set(mol_list)
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    for thre in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1,max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print('thre', thre, 'total num of compounds', len(ligand_uniqueness), 'num of clusters', max(C_clusters), 'max length', max(len_list))
        C_cluster_dict = {ligand_uniqueness[i]:C_clusters[i] for i in range(len(ligand_uniqueness))}
        with open('./PDBBindv2021_similarity/compound_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)


def protein_sim_calculate(protein_unique_list):
    protein_data = pd.read_csv(f'DDE_features_2021.csv', header=None)
    protein_features = protein_data.iloc[:, 1:].values
    protein_sim_mat = cosine_similarity(protein_features).tolist()
    protein_sim_mat = np.array(protein_sim_mat, dtype=np.float32)
    return protein_sim_mat
    # np.save(f'../preprocessing/pdbbind_protein_sim_mat_{MEASURE}.npy', protein_sim_mat)
    # print(f'protein sim_mat_{MEASURE}', protein_sim_mat.shape)


def protein_clustering():
    print('start protein clustering...')
    df = pd.read_csv('./pdbbind_data_last.csv')
    unique_sequences = set(df['Sequence'])
    protein_unique_list = list(unique_sequences)
    print("protein uniqueness: ",len(protein_unique_list))

    protein_sim_mat = protein_sim_calculate(protein_unique_list)
    # protein_sim_mat = np.load(f'../preprocessing/pdbbind_protein_sim_mat_{MEASURE}.npy')
    print('protein sim_mat_2021', protein_sim_mat.shape)

    P_dist = []
    for i in range(protein_sim_mat.shape[0]):
        P_dist += (1-protein_sim_mat[i,(i+1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    for thre in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        P_clusters = fcluster(P_link, thre, 'distance')
        len_list = []
        for i in range(1,max(P_clusters)+1):
            len_list.append(P_clusters.tolist().count(i))
        print('thre', thre, 'total num of proteins', len(protein_unique_list), 'num of clusters', max(P_clusters), 'max length', max(len_list))
        print(len(protein_unique_list), len(P_clusters))
        P_cluster_dict = {protein_unique_list[i]:P_clusters[i] for i in range(len(protein_unique_list))}
        with open('./PDBBindv2021_similarity/protein_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)

df = pd.read_csv('pdbbind_data_last.csv')
# pdbbind_id = df['PDBname']
smile = df['Smile']
compound_clustering(smile)
protein_clustering()


