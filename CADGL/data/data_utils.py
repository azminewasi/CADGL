import pandas as pd 
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import numpy as np
from tdc.chem_utils import featurize
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors, rdFreeSASA
from rdkit.Chem.rdFreeSASA import CalcSASA

from data.structures import generate_feature_matrix


def convert_to_hetero_graph(df, drug_2_smiles, drug_2_idx, relation_id):
    data = pyg_data.HeteroData()
    
    drug_name = list(drug_2_idx.keys())

    morgan = generate_feature_matrix(drug_2_smiles, drug_2_idx)

    morgan_array = morgan.toarray()
 
    data["drug"].x = torch.from_numpy(morgan_array).float()

    df["d1"] = df["d1"].apply(lambda row: drug_2_idx[row]).to_numpy()
    df["d2"] = df["d2"].apply(lambda row: drug_2_idx[row]).to_numpy()
    
    edge_index_dict = {}
    
    for id_ in relation_id:
        edge_df = df[df.type == id_]
        d1 = edge_df["d1"]
        d2 = edge_df["d2"]
        edges = np.stack([d1, d2], axis = 0)
        edge_index_dict[str(id_)] = pyg_utils.to_undirected(
                                        torch.from_numpy(edges).long())
        
        data["drug", str(id_), "drug"].edge_index = pyg_utils.to_undirected(
                                                torch.from_numpy(edges).long())
    return data
    
def generate_label(df, drug_2_idx, relation_id, device = "cpu"):
    edge_label_index_dict = {}
    edge_label_dict = {}

    try:
        df["d1"] = df["d1"].apply(lambda row: drug_2_idx[row]).to_numpy()
        df["d2"] = df["d2"].apply(lambda row: drug_2_idx[row]).to_numpy()
    except:
        pass 
    
    for id_ in relation_id:
        edge_df = df[df.type == id_]
        d1 = edge_df["d1"]
        d2 = edge_df["d2"]
        edges = np.stack([d1, d2], axis = 0)
        pos_labels = np.zeros(edges.shape[1])
        edge_label_index_dict[str(id_)] = torch.from_numpy(edges).long().to(device)
        edge_label_dict[str(id_)] = torch.from_numpy(pos_labels).float().to(device)
    
    return edge_label_index_dict, edge_label_dict

def count_num_labels(label_dict):
    return sum(label_dict[i].shape[0] for i in label_dict.keys())