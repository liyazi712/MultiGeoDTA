"""
Drug-target binding affinity datasets
"""
import os
import torch
from torch_geometric.data import Batch
import numpy as np
import pandas as pd
import torch.utils.data as data
from MultiGeoDTA import pdb_graph, mol_graph
import gzip
import pickle


def load_dict(input_path):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)

def save_dict(mol_dict, output_path):
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(mol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

class DTA(data.Dataset):
    """
    Base class for loading drug-target binding affinity datasets.
    Adapted from: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py

    This class is used to train model, where labels is needed.
    """
    def __init__(self, data_list=None):

        super(DTA, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.data_list[idx]['y'] is not None:
            drug = self.data_list[idx]['drug_graph']
            prot = self.data_list[idx]['protein_graph']
            full_seq = self.data_list[idx]['full_sequence']
            poc_seq = self.data_list[idx]['pocket_sequence']
            smile_encode = self.data_list[idx]['smile_sequence']
            smile = self.data_list[idx]['smile']
            y = self.data_list[idx]['y']

            return drug, prot, full_seq, poc_seq, smile_encode, float(y), smile

        else:
            drug = self.data_list[idx]['drug_graph']
            prot = self.data_list[idx]['protein_graph']
            full_seq = self.data_list[idx]['full_sequence']
            poc_seq = self.data_list[idx]['pocket_sequence']
            smile_encode = self.data_list[idx]['smile_sequence']
            smile = self.data_list[idx]['smile']

            return drug, prot, full_seq, poc_seq, smile_encode, smile

    def collate(self, sample):
        # batch_size = len(sample)
        # print(f'batch_size: {batch_size}')
        if len(map(list, zip(*sample))) == 7:
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, label, smile = map(list, zip(*sample))
            compound_graph = Batch.from_data_list(compound_graph)
            protein_graph = Batch.from_data_list(protein_graph)
            full_seq = torch.tensor(np.array(full_seq)).long()
            poc_seq = torch.tensor(np.array(poc_seq)).long()
            smile_seq = torch.tensor(np.array(smile_seq)).long()
            SMILE = list(smile)
            label = torch.FloatTensor(label)

            item = {'drug': compound_graph, 'protein': protein_graph, 'full_seq': full_seq, 'poc_seq': poc_seq,
                    'smile_seq': smile_seq, 'y': label, 'SMILES': SMILE}
            return item

        elif len(map(list, zip(*sample))) == 6:
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, smile = map(list, zip(*sample))
            compound_graph = Batch.from_data_list(compound_graph)
            protein_graph = Batch.from_data_list(protein_graph)
            full_seq = torch.tensor(np.array(full_seq)).long()
            poc_seq = torch.tensor(np.array(poc_seq)).long()
            smile_seq = torch.tensor(np.array(smile_seq)).long()
            SMILE = list(smile)

            item = {'drug': compound_graph, 'protein': protein_graph, 'full_seq': full_seq, 'poc_seq': poc_seq,
                    'smile_seq': smile_seq, 'SMILES': SMILE}
            return item

        else:
            raise ValueError(
                f"Unexpected number of elements in sample. Expected 6 or 7, but got {len(map(list, zip(*sample)))}.")


# class DTA_inference(data.Dataset):
#     """
#     If you wang to use this model to predict new data, this class should be used. (Don't need to provide labels)
#     """
#     def __init__(self, data_list=None):
#
#         super(DTA_inference, self).__init__()
#         self.data_list = data_list
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, idx):
#         drug = self.data_list[idx]['drug_graph']
#         prot = self.data_list[idx]['protein_graph']
#         full_seq = self.data_list[idx]['full_sequence']
#         poc_seq = self.data_list[idx]['pocket_sequence']
#         smile_encode = self.data_list[idx]['smile_sequence']
#         smile = self.data_list[idx]['smile']
#
#         return drug, prot, full_seq, poc_seq, smile_encode, smile
#
#     def collate(self, sample):
#         # batch_size = len(sample)
#         # print(f'batch_size: {batch_size}')
#
#         compound_graph, protein_graph, full_seq, poc_seq, smile_seq, smile = map(list, zip(*sample))
#         compound_graph = Batch.from_data_list(compound_graph)
#         protein_graph = Batch.from_data_list(protein_graph)
#         full_seq = torch.tensor(np.array(full_seq)).long()
#         poc_seq = torch.tensor(np.array(poc_seq)).long()
#         smile_seq = torch.tensor(np.array(smile_seq)).long()
#         SMILE = list(smile)
#
#         item = {'drug': compound_graph, 'protein': protein_graph, 'full_seq': full_seq, 'poc_seq': poc_seq,
#                 'smile_seq': smile_seq, 'SMILES': SMILE}
#         return item

class DTATask(object):
    """
    Drug-target affinity task.
    """
    def __init__(self,
            task_name=None,
            split_method = None,
            train_data=None, valid_data=None, test_data=None,
            train_pdb_data=None, valid_pdb_data=None, test_pdb_data=None,
            train_sdf_data=None, valid_sdf_data=None, test_sdf_data=None,
            max_seq_len=None, max_smi_len=None,
            emb_dir=None,
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8., onthefly=False
        ):

        self.task_name = task_name
        self.train_pdb_data = train_pdb_data
        self.valid_pdb_data = valid_pdb_data
        self.test_pdb_data = test_pdb_data
        self.train_sdf_data = train_sdf_data
        self.valid_sdf_data = valid_sdf_data
        self.test_sdf_data = test_sdf_data
        self.emb_dir = emb_dir
        self.split_method = split_method
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

        self.prot_featurize_params = dict(
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff)
        self._prot2pdb = None
        self._pdb_graph_db = None        
        self._drug2sdf_file = None
        self._drug_sdf_db = None
        self.onthefly = onthefly

    def _format_pdb_entry(self, _data):
        _coords = _data["coords"]
        entry = {
            "seq": _data["seq"],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
        }
        # If protein language model embedding(such as ESM-C, ProtTrans) is provided, this will load it.
        # But in this work,we don't use it. So self.emb_dir is None
        if self.emb_dir is not None:
            embed_file = f"{_data['PDB_id']}.{_data['chain']}.pt"
            entry["embed"] = f"{self.emb_dir}/{embed_file}"
        return entry

    @property
    def pdb_graph_db(self):
        if self._pdb_graph_db is None:
            pdbid_entry = {}
            # Training phase, where dataset is split into training, valid, test dataset
            if self.test_pdb_data is not None and self.valid_pdb_data is not None:
                combined_pdb_data = {**self.train_pdb_data, **self.valid_pdb_data, **self.test_pdb_data}
            # Inference phase, where only test dataset is provided.
            else:
                combined_pdb_data = {**self.test_pdb_data}

            for pdb_id, protein_structure_info in combined_pdb_data.items():
                pdbid_entry[pdb_id] = self._format_pdb_entry(protein_structure_info)
            self._pdb_graph_db = pdb_graph.pdb_to_graphs(pdbid_entry, self.prot_featurize_params)
        return self._pdb_graph_db

    @property
    def drug_sdf_db(self):
        if self._drug_sdf_db is None:
            # Training phase, where dataset is split into training, valid, test dataset
            if self.train_sdf_data is not None and self.valid_sdf_data is not None:
                combined_sdf_data = {**self.train_sdf_data, **self.valid_sdf_data, **self.test_sdf_data}
            # Inference phase, where only test dataset is provided.
            else:
                combined_sdf_data = {**self.test_sdf_data}

            self._drug_sdf_db = mol_graph.sdf_to_graphs(combined_sdf_data)
        return self._drug_sdf_db

    def label_seq(self, line, max_seq_len):
        protein_char = {
            '<MASK>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
            'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
            'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
            'W': 19, 'Y': 20, 'U': 21, 'O': 22, 'B': 23, 'Z': 24, 'J': 25, 'X': 26
                        }

        label = np.zeros(max_seq_len)
        for i, lab in enumerate(line[:max_seq_len]):
            label[i] = protein_char[lab]
        return label

    def smiles2onehot(self, smiles):
        CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                         "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                         "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                         "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                         "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                         "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                         "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                         "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ":": 0, "~": 65}

        x = np.zeros(self.max_smi_len)

        for i, ch in enumerate(smiles[:self.max_smi_len]):
            x[i] = CHARISOSMISET.get(ch, 66) # Default is 66

        return x

    def position_seq(self, seq, position):
        res = ['<MASK>'] * len(seq)
        for i in position:
            res[i - 1] = seq[i - 1]
        return res


    def build_data_when_training(self, data, split, save_processed_file):
        records = data.to_dict('records')
        # print(records)
        data_list = []
        if os.path.exists(f'./create_dataset/{self.task_name}/processed_data_dict_{split}.pkl.gz'):
            with gzip.open(f'./create_dataset/{self.task_name}/processed_data_dict_{split}.pkl.gz', 'rb') as f:
                data_list = pickle.load(f)
            print(f"{split} data is loaded from file!")
        else:
            print(f"File not found, starting to process {split} data from scratch.")
            for entry in records:
                # print(entry)
                pdb_id = entry['PDBname']
                df = self.drug_sdf_db[pdb_id]
                pf = self.pdb_graph_db[pdb_id]
                seq = entry['Sequence']
                position = eval(entry['Position'])
                smile = entry['Smile']

                smile_encode = self.smiles2onehot(smile)
                pocket = self.position_seq(seq, position)
                seq_encode = self.label_seq(seq, self.max_seq_len)
                pocket_encode = self.label_seq(pocket, self.max_seq_len)

                data_list.append({
                    'drug_graph': df,
                    'protein_graph': pf,
                    'full_sequence': seq_encode,
                    'pocket_sequence': pocket_encode,
                    'smile_sequence': smile_encode,
                    'y': entry['label'],
                    'pdb_name': pdb_id,
                    'smile': smile
                })
            # 保存数据到文件
            if save_processed_file:
                with gzip.open(f'./create_dataset/{self.task_name}/processed_data_dict_{split}.pkl.gz', 'wb') as f:
                    pickle.dump(data_list, f)
                print(f"{split} data is saved in the file!")

        data = DTA(data_list=data_list)
        return data

    # with dataset without label
    def build_data_when_inference(self, data, save_processed_file):
        records = data.to_dict('records')
        data_list = []
        if os.path.exists(f'./create_dataset/zinc/processed_data_dict_new.pkl.gz'):
            with gzip.open(f'./create_dataset/zinc/processed_data_dict_new.pkl.gz', 'rb') as f:
                data_list = pickle.load(f)
            print("Data is loaded from file!")
        else:
            print("File not found, starting to process data from scratch.")
            for entry in records:
                pdb_id = entry['zinc_id']
                pf = self.pdb_graph_db['target']
                df = self.drug_sdf_db[pdb_id]

                # When virtual screening phase, you can replace 'seq' and 'position' manually

                # protein_sequence_CB1R
                seq = 'GENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLAAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDIFPHIDKTYLMFWIGVVSVLLLFIVYAYMYILWKAHSHAVAKALIVYGSTTGNTEYTAETIARELADAGYEVDSRDAASVEAGGLFEGFDLVLLGCSTWGDDSIELQDDFIPLFDSLEETGAQGRKVACFGCGDSSWEYFCGAVDAIEEKLKNLGAEIVQDGLRIDGDPRAARDDIVGWAHDVRGAIPDQARMDIELAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPS'
                # protein_sequence_HCAR1
                # seq = 'CCRIEGDTISQVMPPLLIVAFVLGALGNGVALCGFCFHMKTWKPSTVYLFNLAVADFLLMICLPFRTDYYLRRRHWAFGDIPCRVGLFTLAMNRAGSIVFLTVVAADRYFKVVHPHHAVNTISTRVAAGIVCTLWALVILGTVYLLLENHLCVQETAVSCESFIMESANGWHDIMFQLEFFMPLGIILFCSFKIVWSLRRRQQLARQARMKKATRFIMVVAIVFITCYLPSVSARLYFLWTVPSSACDPSVHGALHITLSFTYMNSMLDPLVYYFSSPSLDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLVSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWNTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMTGIVETHFTFKDLHFKMFDVGAQRSERKKWIHCFEGVTAIIFCVALSDYDLVNRMHESMKLFDSICNNKWFTDTSIILFLNKKDLFEEKIKKSPLTICYPEYAGSNTYEEAAAYIQCQFEDLNKRKDTKEIYTHFTCSTDTKNVQFVFDAVTDVIIKNNLKDCGLFSIAQARKLVEQLKMEANIDRIKVSKAAADLMAYCEAHAKEDPLLTPVPASENPFREDVQLVESGGGLVQPGGSRKLSCSASGFAFSSFGMHWVRQAPEKGLEWVAYISSGSGTIYYADTVKGRFTISRDDPKNTLFLQMTSLRSEDTAMYYCVRSIYYYGSSPFDFWGQGTTLTVSSSDIVMTQATSSVPVTPGESVSISCRSSKSLLHSNGNTYLYWFLQRPGQSPQLLIYRMSNLASGVPDRFSGSGSGTAFTLTISRLEAEDVGVYYCMQHLEYPLTFGAGTKLEL'

                # pocket_position_CB1R
                position = [10, 12, 72, 75, 76, 79, 80, 86, 91, 94, 95, 98, 99, 102, 168, 169, 170, 171, 173, 177, 178, 181, 380, 383, 387, 400, 401, 403, 404, 407, 410]
                # pocket_position_HCAR1
                # position = [9, 13, 14, 16, 17, 58, 59, 62, 63, 64, 65, 66, 67, 69, 70, 74, 83, 84, 85, 86, 87, 88, 89,
                #             90, 91, 92, 93, 94, 95, 142, 144, 145, 150, 159, 160, 161, 162, 163, 164, 165, 172, 175,
                #             176, 179, 228, 231, 235, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 266]
                smile = entry['SMILES']
                smile_encode = self.smiles2onehot(smile)
                pocket = self.position_seq(seq, position)
                seq_encode = self.label_seq(seq, self.max_seq_len)
                pocket_encode = self.label_seq(pocket, self.max_seq_len)

                data_list.append({
                    'drug_graph': df,
                    'protein_graph': pf,
                    'full_sequence': seq_encode,
                    'pocket_sequence': pocket_encode,
                    'smile_sequence': smile_encode,
                    'pdb_name': pdb_id,
                    'smile': smile
                })

            if save_processed_file:
                with gzip.open('./create_dataset/zinc/processed_data_dict_new.pkl.gz', 'wb') as f:
                    pickle.dump(data_list, f)
                print("Data is saved in the file!")

        data = DTA(data_list=data_list)
        return data

    def get_split(self):
        split_df = {'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data}
        split_data = {}
        # Training phase, where dataset is split into training, valid, test dataset
        if self.train_data is not None and self.valid_data is not None:
            split_data['train'] = self.build_data_when_training(self.train_data, split='train', save_processed_file=True)
            split_data['valid'] = self.build_data_when_training(self.valid_data, split='valid', save_processed_file=True)
            split_data['test'] = self.build_data_when_training(self.test_data, split='test', save_processed_file=True)
            print("train_dataset size: ", len(split_data['train']), "valid_dataset size: ", len(split_data['valid']),
                  "test_dataset size: ", len(split_data['test']))
        # Inference phase, where only test dataset is provided.
        else:
            # zinc dataset
            split_data['test'] = self.build_data_when_inference(self.test_data, save_processed_file=True)
            print("Only test dataset! test_dataset size: ", len(split_data['test']))

        return split_data, split_df

class pdbbind_v2016(DTATask):
    """
    pdbbind_v2016 drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/pdbbind_v2016/last_train_2016.csv',
            valid_path='./create_dataset/pdbbind_v2016/last_valid_2016.csv',
            test_path='./create_dataset/pdbbind_v2016/last_test_2016.csv',
            train_pdb='./create_dataset/pdbbind_v2016/pocket_structures_train.pkl.gz',
            valid_pdb='./create_dataset/pdbbind_v2016/pocket_structures_valid.pkl.gz',
            test_pdb='./create_dataset/pdbbind_v2016/pocket_structures_test.pkl.gz',
            train_sdf='./create_dataset/pdbbind_v2016/mol_structures_train.pkl.gz',
            valid_sdf='./create_dataset/pdbbind_v2016/mol_structures_valid.pkl.gz',
            test_sdf='./create_dataset/pdbbind_v2016/mol_structures_test.pkl.gz',
            num_pos_emb=16, num_rbf=16, max_seq_len=1024, max_smi_len=128,
            contact_cutoff=8.,
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = load_dict(train_pdb)
        valid_pdb_data = load_dict(valid_pdb)
        test_pdb_data = load_dict(test_pdb)
        train_sdf_data = load_dict(train_sdf)
        valid_sdf_data = load_dict(valid_sdf)
        test_sdf_data = load_dict(test_sdf)
        super(pdbbind_v2016, self).__init__(
            task_name='pdbbind_v2016',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
            )


class pdbbind_v2020(DTATask):
    """
    pdbbind_v2020 drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/pdbbind_v2020/last_train_2020.csv',
            valid_path='./create_dataset/pdbbind_v2020/last_valid_2020.csv',
            test_path='./create_dataset/pdbbind_v2020/last_test_2020.csv',
            train_pdb='./create_dataset/pdbbind_v2020/pocket_structures_train.pkl.gz',
            valid_pdb='./create_dataset/pdbbind_v2020/pocket_structures_valid.pkl.gz',
            test_pdb='./create_dataset/pdbbind_v2020/pocket_structures_test.pkl.gz',
            train_sdf='./create_dataset/pdbbind_v2020/mol_structures_train.pkl.gz',
            valid_sdf='./create_dataset/pdbbind_v2020/mol_structures_valid.pkl.gz',
            test_sdf='./create_dataset/pdbbind_v2020/mol_structures_test.pkl.gz',
            num_pos_emb=16, num_rbf=16, max_seq_len=1024, max_smi_len=128,
            contact_cutoff=8.,
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = load_dict(train_pdb)
        valid_pdb_data = load_dict(valid_pdb)
        test_pdb_data = load_dict(test_pdb)
        train_sdf_data = load_dict(train_sdf)
        valid_sdf_data = load_dict(valid_sdf)
        test_sdf_data = load_dict(test_sdf)
        super(pdbbind_v2020, self).__init__(
            task_name='pdbbind_v2020',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
            )

class pdbbind_v2021_time(DTATask):
    """
    pdbbind_v2021_time drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/pdbbind_v2021_time/train_2021.csv',
            valid_path='./create_dataset/pdbbind_v2021_time/valid_2021.csv',
            test_path='./create_dataset/pdbbind_v2021_time/test_2021.csv',
            train_pdb='./create_dataset/pdbbind_v2021_time/pocket_structures_train.pkl.gz',
            valid_pdb='./create_dataset/pdbbind_v2021_time/pocket_structures_valid.pkl.gz',
            test_pdb='./create_dataset/pdbbind_v2021_time/pocket_structures_test.pkl.gz',
            train_sdf='./create_dataset/pdbbind_v2021_time/mol_structures_train.pkl.gz',
            valid_sdf='./create_dataset/pdbbind_v2021_time/mol_structures_valid.pkl.gz',
            test_sdf='./create_dataset/pdbbind_v2021_time/mol_structures_test.pkl.gz',
            num_pos_emb=16, num_rbf=16, max_seq_len=1024, max_smi_len=128,
            contact_cutoff=8.,
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = load_dict(train_pdb)
        valid_pdb_data = load_dict(valid_pdb)
        test_pdb_data = load_dict(test_pdb)
        train_sdf_data = load_dict(train_sdf)
        valid_sdf_data = load_dict(valid_sdf)
        test_sdf_data = load_dict(test_sdf)
        super(pdbbind_v2021_time, self).__init__(
            task_name='pdbbind_v2021_time',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
            )

class pdbbind_v2021_similarity(DTATask):
    """
    pdbbind_v2021_similarity drug-target interaction dataset
    """
    def __init__(self,
            setting='new_new', thre=0.5,
            num_pos_emb=16, num_rbf=16, max_seq_len=800, # 1024
            max_smi_len=256, # 128
            contact_cutoff=8.,
        ):

        train_path = f'./create_dataset/pdbbind_v2021_similarity/{setting}/train_{thre}.csv'
        valid_path = f'./create_dataset/pdbbind_v2021_similarity/{setting}/valid_{thre}.csv'
        test_path = f'./create_dataset/pdbbind_v2021_similarity/{setting}/test_{thre}.csv'
        train_pdb = f'./create_dataset/pdbbind_v2021_similarity/{setting}/pocket_structures_train_{thre}.pkl.gz'
        valid_pdb = f'./create_dataset/pdbbind_v2021_similarity/{setting}/pocket_structures_valid_{thre}.pkl.gz'
        test_pdb = f'./create_dataset/pdbbind_v2021_similarity/{setting}/pocket_structures_test_{thre}.pkl.gz'
        train_sdf = f'./create_dataset/pdbbind_v2021_similarity/{setting}/mol_structures_train_{thre}.pkl.gz'
        valid_sdf = f'./create_dataset/pdbbind_v2021_similarity/{setting}/mol_structures_valid_{thre}.pkl.gz'
        test_sdf = f'./create_dataset/pdbbind_v2021_similarity/{setting}/mol_structures_test_{thre}.pkl.gz'

        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = load_dict(train_pdb)
        valid_pdb_data = load_dict(valid_pdb)
        test_pdb_data = load_dict(test_pdb)
        train_sdf_data = load_dict(train_sdf)
        valid_sdf_data = load_dict(valid_sdf)
        test_sdf_data = load_dict(test_sdf)
        super(pdbbind_v2021_similarity, self).__init__(
            task_name='pdbbind_v2021_similarity',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
            )


class lp_pdbbind(DTATask):
    """
    lp_pdbbind drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/lp_pdbbind/LP_PDBBind_train.csv',
            valid_path='./create_dataset/lp_pdbbind/LP_PDBBind_valid.csv',
            test_path='./create_dataset/lp_pdbbind/LP_PDBBind_test.csv',
            train_pdb='./create_dataset/lp_pdbbind/pocket_structures_train.pkl.gz',
            valid_pdb='./create_dataset/lp_pdbbind/pocket_structures_valid.pkl.gz',
            test_pdb='./create_dataset/lp_pdbbind/pocket_structures_test.pkl.gz',
            train_sdf='./create_dataset/lp_pdbbind/mol_structures_train.pkl.gz',
            valid_sdf='./create_dataset/lp_pdbbind/mol_structures_valid.pkl.gz',
            test_sdf='./create_dataset/lp_pdbbind/mol_structures_test.pkl.gz',
            num_pos_emb=16, num_rbf=16, max_seq_len=1024,
            contact_cutoff=8., max_smi_len=128,
        ):
        train_data = pd.read_csv(train_path, index_col=0)
        valid_data = pd.read_csv(valid_path, index_col=0)
        test_data = pd.read_csv(test_path, index_col=0)
        train_pdb_data = load_dict(train_pdb)
        valid_pdb_data = load_dict(valid_pdb)
        test_pdb_data = load_dict(test_pdb)
        train_sdf_data = load_dict(train_sdf)
        valid_sdf_data = load_dict(valid_sdf)
        test_sdf_data = load_dict(test_sdf)
        super(lp_pdbbind, self).__init__(
            task_name='lp_pdbbind',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
            )


class zinc(DTATask):
    """
    lp_pdbbind drug-target interaction dataset
    """
    def __init__(self,
            test_path='./create_dataset/zinc/processed_zinc_CB1R.csv',
            test_pdb='./create_dataset/zinc/pocket_structures.pkl.gz',
            test_sdf='./create_dataset/zinc/mol_structures.pkl.gz',
            num_pos_emb=16, num_rbf=16, max_seq_len=1024, max_smi_len=128,
            contact_cutoff=8.,
        ):
        test_data = pd.read_csv(test_path)
        test_pdb_data = load_dict(test_pdb)
        test_sdf_data = load_dict(test_sdf)
        super(zinc, self).__init__(
            task_name='zinc',
            test_data=test_data,
            test_pdb_data=test_pdb_data,
            test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len, max_smi_len=max_smi_len,
            contact_cutoff=contact_cutoff,
        )