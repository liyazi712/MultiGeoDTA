import csv
import pickle
from sklearn.model_selection import KFold
import numpy as np
import json
import os
import pandas as pd


def split_train_test_clusters(measure, clu_thre, n_fold):
    # load cluster dict
    # cluster_path = '../preprocessing/'
    with open('./PDBBindv2021_similarity/compound_cluster_dict_' + str(clu_thre), 'rb') as f:
        C_cluster_dict = pickle.load(f)
    with open('./PDBBindv2021_similarity/protein_cluster_dict_' + str(clu_thre), 'rb') as f:
        P_cluster_dict = pickle.load(f)

    C_cluster_set = set(list(C_cluster_dict.values()))
    P_cluster_set = set(list(P_cluster_dict.values()))
    C_cluster_list = np.array(list(C_cluster_set))
    P_cluster_list = np.array(list(P_cluster_set))
    np.random.shuffle(C_cluster_list)
    np.random.shuffle(P_cluster_list)
    c_kf = KFold(n_fold, shuffle=True, random_state=42)
    p_kf = KFold(n_fold, shuffle=True, random_state=42)
    c_train_clusters, c_test_clusters = [], []
    for train_idx, test_idx in c_kf.split(C_cluster_list):
        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])
    p_train_clusters, p_test_clusters = [], []
    for train_idx, test_idx in p_kf.split(P_cluster_list):
        p_train_clusters.append(P_cluster_list[train_idx])
        p_test_clusters.append(P_cluster_list[test_idx])

    pair_list = []
    for i_c in C_cluster_list:
        for i_p in P_cluster_list:
            pair_list.append('c' + str(i_c) + 'p' + str(i_p))
    pair_list = np.array(pair_list)
    np.random.shuffle(pair_list)
    pair_kf = KFold(n_fold, shuffle=True, random_state=42)
    pair_train_clusters, pair_test_clusters = [], []
    for train_idx, test_idx in pair_kf.split(pair_list):
        pair_train_clusters.append(pair_list[train_idx])
        pair_test_clusters.append(pair_list[test_idx])

    return pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def load_data(measure, setting, clu_thre, n_fold):
    df = pd.read_csv('./pdbbind_data_last.csv')
    pid_list = df.iloc[:, 2].tolist()
    cid_list = df.iloc[:, 1].tolist()
    data_pack = df.values.tolist()
    n_sample = len(data_pack)

    # train-test split
    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    print('setting:', setting)
    if setting == 'imputation':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
            = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            pair_train_valid, pair_test = pair_train_clusters[fold], pair_test_clusters[fold]
            pair_valid = np.random.choice(pair_train_valid, int(len(pair_train_valid) * 0.125), replace=False)
            pair_train = set(pair_train_valid) - set(pair_valid)
            pair_valid = set(pair_valid)
            pair_test = set(pair_test)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if 'c' + str(C_cluster_dict[cid_list[ele]]) + 'p' + str(P_cluster_dict[pid_list[ele]]) in pair_train:
                    train_idx.append(ele)
                elif 'c' + str(C_cluster_dict[cid_list[ele]]) + 'p' + str(P_cluster_dict[pid_list[ele]]) in pair_valid:
                    valid_idx.append(ele)
                elif 'c' + str(C_cluster_dict[cid_list[ele]]) + 'p' + str(P_cluster_dict[pid_list[ele]]) in pair_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
        # print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))

    elif setting == 'new_protein':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
            = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
            p_valid = np.random.choice(p_train_valid, int(len(p_train_valid) * 0.125), replace=False)
            p_train = set(p_train_valid) - set(p_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if P_cluster_dict[pid_list[ele]] in p_train:
                    train_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_valid:
                    valid_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
        # print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))

    elif setting == 'new_compound':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
            = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
            c_valid = np.random.choice(c_train_valid, int(len(c_train_valid) * 0.125), replace=False)
            c_train = set(c_train_valid) - set(c_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if C_cluster_dict[cid_list[ele]] in c_train:
                    train_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_valid:
                    valid_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
        # print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))

    elif setting == 'new_new':
        assert n_fold ** 0.5 == int(n_fold ** 0.5)
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
            = split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))

        for fold_x in range(int(n_fold ** 0.5)):
            for fold_y in range(int(n_fold ** 0.5)):
                c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
                c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
                c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid) / 3), replace=False)
                c_train = set(c_train_valid) - set(c_valid)
                p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid) / 3), replace=False)
                p_train = set(p_train_valid) - set(p_valid)

                train_idx, valid_idx, test_idx = [], [], []
                for ele in range(n_sample):
                    if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
                        train_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
                        valid_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
                        test_idx.append(ele)
                train_idx_list.append(train_idx)
                valid_idx_list.append(valid_idx)
                test_idx_list.append(test_idx)
            # print('fold', fold_x*int(n_fold ** 0.5)+fold_y, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
    return data_pack, train_idx_list, valid_idx_list, test_idx_list


measures = ['All']
settings = ['new_compound', 'new_protein', 'new_new']
clu_thres = [0.3, 0.4, 0.5, 0.6]
# clu_thres = [0.1, 0.2, 0.3, 0.4, 0.5]

# 创建一个空的 DataFrame 来存储结果
results_df = pd.DataFrame(columns=['Measure', 'Setting', 'Clu_Thre', 'Fold', 'Train_Num', 'Valid_Num', 'Test_Num'])

for measure in measures:
    for setting in settings:
        for clu_thre in clu_thres:
            if setting == 'new_compound':
                n_fold = 5
            elif setting == 'new_protein':
                n_fold = 5
            elif setting == 'new_new':
                n_fold = 9

            print('Dataset: PDBbind v2021 with measurement', measure)
            print('Clustering threshold:', clu_thre)
            print('dataset split setting:', setting)

            # load data
            data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold)

            for a_fold in range(n_fold):
                print('fold', a_fold + 1, 'begin')
                train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
                print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))
                train_list = [[data_pack[i][j] for i in train_idx] for j in range(0, 8)]
                valid_list = [[data_pack[i][j] for i in valid_idx] for j in range(0, 8)]
                test_list = [[data_pack[i][j] for i in test_idx] for j in range(0, 8)]

                train_df = pd.DataFrame(np.array(train_list)).T
                valid_df = pd.DataFrame(np.array(valid_list)).T
                test_df = pd.DataFrame(np.array(test_list)).T

                row_df = {
                    'Measure': measure,
                    'Setting': setting,
                    'Clu_Thre': clu_thre,
                    'Fold': a_fold + 1,
                    'Train_Num': len(train_idx),
                    'Valid_Num': len(valid_idx),
                    'Test_Num': len(test_idx)
                }

                results_df = pd.concat([results_df, pd.DataFrame([row_df])], ignore_index=True)
                json_dir = f'./PDBBindv2021_similarity/{setting}'

                train_df.to_csv(f'./PDBBindv2021_similarity/{setting}/train_{a_fold + 1}_{clu_thre}.csv',
                                index=False)
                valid_df.to_csv(f'./PDBBindv2021_similarity/{setting}/valid_{a_fold + 1}_{clu_thre}.csv',
                                index=False)
                test_df.to_csv(f'./PDBBindv2021_similarity/{setting}/test_{a_fold + 1}_{clu_thre}.csv',
                               index=False)


# 保存到 CSV 文件
results_df.to_csv('./PDBBindv2021_similarity/dataset_split_info.csv', index=False)
print("Dataset split information saved to './PDBBindv2021_similarity/dataset_split_info.csv'")
