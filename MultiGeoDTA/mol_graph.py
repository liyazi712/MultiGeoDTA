from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import torch_geometric
from MultiGeoDTA.constants import ATOM_VOCAB
from MultiGeoDTA.pdb_graph import _rbf, _normalize


def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]

    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)

    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values

def get_edge_index_from_mol(mol):
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def _build_atom_feature(mol):
    if isinstance(mol, Chem.Mol):
        # 确保分子有明确的立体化学信息
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except:
            pass  # 如果无法赋值立体化学，跳过

    feature_alphabet = {
        'GetSymbol': (ATOM_VOCAB, 'unk'),  # 原子符号
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),  # 原子度数
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),  # 氢原子总数
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),  # 隐式化合价
        'GetIsAromatic': ([0, 1], 1),  # 是否芳香性
        'GetFormalCharge': ([-2, -1, 0, 1, 2], 0),  # 形式电荷
        'GetHybridization': ([Chem.HybridizationType.SP,
                              Chem.HybridizationType.SP2,
                              Chem.HybridizationType.SP3,
                              Chem.HybridizationType.SP3D,
                              Chem.HybridizationType.SP3D2], Chem.HybridizationType.SP3),  # 杂化类型
        'IsInRing': ([0, 1], 1),  # 是否在环中
        # 新增离散特征
        'IsChiral': ([0, 1], 0),  # 是否为手性中心
        'IsHDonor': ([0, 1], 0),  # 是否为氢键供体
        'IsHAcceptor': ([0, 1], 0),  # 是否为氢键受体
    }

    # 连续值特征
    continuous_features = {
        'GetMass': (0, 200),  # 原子质量
        'Electronegativity': (0.5, 4.0),  # 电负性
        'CovalentRadius': (0.3, 2.0),  # 共价半径
        'VanDerWaalsRadius': (1.0, 3.0),  # 范德瓦尔斯半径
        'GetPartialCharge': (-1.0, 1.0),  # 部分电荷
        'ValenceElectrons': (0, 18),  # 价电子数
    }

    atom_feature = None

    # 处理离散特征
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                 'GetImplicitValence', 'GetIsAromatic',
                 'GetFormalCharge', 'GetHybridization', 'IsInRing',
                 'IsChiral', 'IsHDonor', 'IsHAcceptor']:
        if attr == 'IsChiral':
            # 检查是否为 QueryAtom，如果是则返回默认值 0
            feature = []
            for atom in mol.GetAtoms():
                if hasattr(atom, 'IsChiral'):
                    feature.append(1 if atom.IsChiral() else 0)
                else:
                    feature.append(0)  # QueryAtom 或无手性信息时默认为非手性
        elif attr == 'IsHDonor':
            # 简单规则：N、O上带H的原子可能是供体
            feature = [1 if (atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0) else 0
                       for atom in mol.GetAtoms()]
        elif attr == 'IsHAcceptor':
            # 简单规则：N、O可能是受体
            feature = [1 if atom.GetSymbol() in ['N', 'O'] else 0 for atom in mol.GetAtoms()]
        else:
            feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]

        feature = onehot_encoder(feature,
                                 alphabet=feature_alphabet[attr][0],
                                 default=feature_alphabet[attr][1],
                                 drop_first=(attr in ['GetIsAromatic', 'IsInRing', 'IsChiral',
                                                      'IsHDonor', 'IsHAcceptor']))
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)

    # 处理连续特征
    continuous_vals = []
    for attr, (min_val, max_val) in continuous_features.items():
        if attr == 'GetMass':
            feature = [atom.GetMass() for atom in mol.GetAtoms()]
        elif attr == 'Electronegativity':
            en_map = {'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
                      'S': 2.58, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}
            feature = [en_map.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()]
        elif attr == 'CovalentRadius':
            cr_map = {'H': 0.31, 'C': 0.77, 'N': 0.70, 'O': 0.66, 'F': 0.57,
                      'S': 1.05, 'P': 1.07, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39}
            feature = [cr_map.get(atom.GetSymbol(), 1.0) for atom in mol.GetAtoms()]
        elif attr == 'VanDerWaalsRadius':
            vdw_map = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                       'S': 1.80, 'P': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
            feature = [vdw_map.get(atom.GetSymbol(), 1.5) for atom in mol.GetAtoms()]
        elif attr == 'GetPartialCharge':
            # 假设分子已计算部分电荷，否则需预先用 RDKit 计算
            feature = [atom.GetDoubleProp('PartialCharge') if atom.HasProp('PartialCharge')
                       else 0.0 for atom in mol.GetAtoms()]
        elif attr == 'ValenceElectrons':
            ve_map = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'S': 6, 'P': 5,
                      'Cl': 7, 'Br': 7, 'I': 7}
            feature = [ve_map.get(atom.GetSymbol(), 4) for atom in mol.GetAtoms()]

        # 归一化到 [0, 1]
        feature = [(x - min_val) / (max_val - min_val) for x in feature]
        continuous_vals.append(np.array(feature).reshape(-1, 1))

    # 合并连续特征
    if continuous_vals:
        continuous_feature = np.concatenate(continuous_vals, axis=1)
        atom_feature = np.concatenate((atom_feature, continuous_feature), axis=1)

    atom_feature = atom_feature.astype(np.float32)
    return atom_feature

def _build_edge_feature(mol, coords, edge_index, D_max=4.5, num_rbf=16):
    """
    参数:
        mol: RDKit Mol 对象
        coords: 原子坐标 (torch.Tensor, shape: [n_atoms, 3])
        edge_index: 边索引 (torch.Tensor, shape: [2, n_edges])
        D_max: RBF 最大距离
        num_rbf: RBF 基函数数量
    返回:
        edge_s: 标量边特征 (torch.Tensor, shape: [n_edges, n_scalar_features])
        edge_v: 向量边特征 (torch.Tensor, shape: [n_edges, n_vector_features, 3])
    """
    # 原始特征：距离向量和 RBF
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    distances = E_vectors.norm(dim=-1)
    rbf = _rbf(distances, D_max=D_max, D_count=num_rbf)
    edge_v = _normalize(E_vectors).unsqueeze(-2)  # [n_edges, 1, 3]

    # 初始化标量特征列表
    edge_s_features = [rbf]  # 初始包含 RBF

    # 电负性映射表
    en_map = {'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
              'S': 2.58, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}

    # 标准键长映射表（单位：Å，近似值）
    bond_length_map = {
        (Chem.BondType.SINGLE, 'C', 'C'): 1.54,
        (Chem.BondType.SINGLE, 'C', 'N'): 1.47,
        (Chem.BondType.SINGLE, 'C', 'O'): 1.43,
        (Chem.BondType.DOUBLE, 'C', 'C'): 1.34,
        (Chem.BondType.DOUBLE, 'C', 'O'): 1.20,
        (Chem.BondType.TRIPLE, 'C', 'C'): 1.20,
    }

    # 构建键特征
    bond_types = []
    is_conjugated = []
    is_in_ring = []
    distance_deviations = []
    electronegativity_diffs = []

    for i in range(edge_index.shape[1]):
        idx0, idx1 = edge_index[0, i], edge_index[1, i]
        atom0, atom1 = mol.GetAtomWithIdx(int(idx0)), mol.GetAtomWithIdx(int(idx1))
        bond = mol.GetBondBetweenAtoms(int(idx0), int(idx1))

        # 键类型
        if bond is not None:
            bond_type = bond.GetBondType()
            bond_types.append([
                bond_type == Chem.BondType.SINGLE,
                bond_type == Chem.BondType.DOUBLE,
                bond_type == Chem.BondType.TRIPLE,
                bond_type == Chem.BondType.AROMATIC
            ])
            is_conjugated.append(bond.GetIsConjugated())
            is_in_ring.append(bond.IsInRing())
        else:
            # 如果没有化学键（例如距离定义的边）
            bond_types.append([1, 0, 0, 0])  # 默认单键
            is_conjugated.append(False)
            is_in_ring.append(False)

        # 电负性差
        en0 = en_map.get(atom0.GetSymbol(), 2.5)
        en1 = en_map.get(atom1.GetSymbol(), 2.5)
        electronegativity_diffs.append(abs(en0 - en1))

        # 键长偏差
        if bond is not None:
            key = (bond.GetBondType(), atom0.GetSymbol(), atom1.GetSymbol())
            key_reverse = (bond.GetBondType(), atom1.GetSymbol(), atom0.GetSymbol())
            std_length = bond_length_map.get(key, bond_length_map.get(key_reverse, 1.5))
            actual_length = distances[i].item()
            distance_deviations.append((actual_length - std_length) / std_length)  # 归一化偏差
        else:
            distance_deviations.append(0.0)  # 无键时偏差为 0

    # 转换为张量
    bond_types = torch.tensor(bond_types, dtype=torch.float32)  # [n_edges, 4]
    is_conjugated = torch.tensor(is_conjugated, dtype=torch.float32).unsqueeze(-1)  # [n_edges, 1]
    is_in_ring = torch.tensor(is_in_ring, dtype=torch.float32).unsqueeze(-1)  # [n_edges, 1]
    distance_deviations = torch.tensor(distance_deviations, dtype=torch.float32).unsqueeze(-1)  # [n_edges, 1]
    electronegativity_diffs = torch.tensor(electronegativity_diffs, dtype=torch.float32).unsqueeze(-1)  # [n_edges, 1]

    # 合并标量特征
    edge_s_features.extend([bond_types, is_conjugated, is_in_ring, distance_deviations, electronegativity_diffs])
    edge_s = torch.cat(edge_s_features, dim=-1)  # [n_edges, num_rbf + 4 + 1 + 1 + 1 + 1]

    edge_v_features = []

    # 1. 归一化方向向量（原始特征）
    edge_v_features.append(_normalize(E_vectors).unsqueeze(-2))  # [n_edges, 1, 3]

    # 2. 原始距离向量（未归一化）
    edge_v_features.append(E_vectors.unsqueeze(-2))  # [n_edges, 1, 3]

    # 3. 电负性梯度向量
    en_map = {'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
              'S': 2.58, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}
    en_diff_vectors = []
    for i in range(edge_index.shape[1]):
        idx0, idx1 = edge_index[0, i], edge_index[1, i]
        atom0, atom1 = mol.GetAtomWithIdx(int(idx0)), mol.GetAtomWithIdx(int(idx1))
        en0 = en_map.get(atom0.GetSymbol(), 2.5)
        en1 = en_map.get(atom1.GetSymbol(), 2.5)
        # 电负性差乘以方向向量
        en_diff_vector = E_vectors[i] * (en1 - en0) / distances[i]  # 方向与极性结合
        en_diff_vectors.append(en_diff_vector)
    edge_v_features.append(torch.stack(en_diff_vectors).unsqueeze(-2))  # [n_edges, 1, 3]

    edge_v = torch.cat(edge_v_features, dim=-2)  # [n_edges, 5, 3]

    # 处理 NaN 值
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(data_list):
    """
    Parameters
    ----------
    data_list: dict, drug key -> sdf file path
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. drug key -> graph
    """
    graphs = {}
    for key, value in tqdm(data_list.items(), desc='sdf'):
        graphs[key] = featurize_drug(mol=value['mol'], coords=value['coords'], atom_type=value['atom_types'], name=key)
    return graphs


def featurize_drug(mol, coords, atom_type, name=None, edge_cutoff=4.5, num_rbf=16):

    with torch.no_grad():
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = _build_atom_feature(mol).tolist()
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        # molecular_graph_edge
        edge_index = get_edge_index_from_mol(mol)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    edge_s, edge_v = _build_edge_feature(mol, coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=atom_type, edge_index=edge_index, name=name,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)
    return data

