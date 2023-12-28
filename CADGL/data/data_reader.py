import pandas as pd 
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import numpy as np
from tdc.chem_utils import featurize
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors, rdFreeSASA
from rdkit.Chem.rdFreeSASA import CalcSASA

def read_files():
    train_df = pd.read_csv("./data/dataset/ddi_training.csv")
    valid_df = pd.read_csv("./data/dataset/ddi_validation.csv")
    test_df = pd.read_csv("./data/dataset/ddi_test.csv")

    drug_smiles = pd.read_csv("./data/dataset/drug_smiles.csv")
    relation_info = pd.read_csv("./data/dataset/Interaction_information.csv")

    drugs = drug_smiles["drug_id"].tolist()
    smiles = drug_smiles["smiles"].tolist()

    drug_2_smiles = {
        drug : smi for (drug, smi) in zip(drugs, smiles)
    }

    drugs.sort()

    drug_2_idx = {
        drug : idx for (drug, idx) in zip(drugs, range(len(drugs)))
    }

    relation_id = relation_info["DDI type"].apply(lambda x: int(x.split()[-1]) - 1).tolist()

    return train_df, valid_df, test_df, drug_2_smiles, drug_2_idx, relation_id