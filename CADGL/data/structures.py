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


def generate_feature_matrix(stitch_2_smile, stitch_2_idx):
    num_drugs = len(stitch_2_idx)
    x = sp.identity(num_drugs, dtype=int, format='csr')
    
    # Physicochemical properties
    mw = [Descriptors.MolWt(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    logp = [Descriptors.MolLogP(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    tpsa = [Descriptors.TPSA(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    
    # Substructure counts
    sc_count = [GraphDescriptors.BalabanJ(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    rings = [rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    
    # Element Wise atom types
    atom_types = []
    for _, smile in stitch_2_smile.items():
        mol = Chem.MolFromSmiles(smile)
        elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        atom_type = [1 if atom.GetSymbol() in elements else 0 for atom in mol.GetAtoms()]
        atom_mean = np.mean(atom_type)
        atom_types.append(atom_mean)

    # ECFP6 fingerprint for each molecule
    ecfp6_fps = []
    for _, smile in stitch_2_smile.items():
        mol = Chem.MolFromSmiles(smile)
        ecfp6_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        ecfp6_arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(ecfp6_fp, ecfp6_arr)
        ecfp6_fps.append(ecfp6_arr)

    # More Feautures: Aromaticity, degree Centrality, Rings, Chirality, Functional Groups
    # charge = [mol.GetFormalCharge() for _, smile in stitch_2_smile.items()]
    aromaticity = [1 if rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(smile)) > 0 else 0 for _, smile in stitch_2_smile.items()]
    logd = [Descriptors.MolLogP(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    degree_centrality = []
    for _, smile in stitch_2_smile.items():
        mol = Chem.MolFromSmiles(smile)
        adj_mat = Chem.rdmolops.GetAdjacencyMatrix(mol)
        degree = np.sum(adj_mat, axis=1)
        degree_centrality.append(np.mean(degree))

    
    # tpsa_sasa_ratio = [Descriptors.TPSA(Chem.MolFromSmiles(smile))/rdFreeSASA.CalcSASA(Chem.MolFromSmiles(smile)) for _, smile in stitch_2_smile.items()]
    chirality = [1 if Chem.FindMolChiralCenters(Chem.MolFromSmiles(smile), includeUnassigned=True) else 0 for _, smile in stitch_2_smile.items()]
    functional_groups = []
    for _, smile in stitch_2_smile.items():
        mol = Chem.MolFromSmiles(smile)
        carboxylic_acid = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')) else 0
        amine = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]')) else 0
        alcohol = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[OH]')) else 0
        ketone = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[C;H2](=O)')) else 0
        ester = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)O')) else 0
        thiol = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[SH]')) else 0
        halogen_bond_acceptor = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[F,Cl,Br,I][#6]')) else 0
        halogen_bond_donor = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[F,Cl,Br,I][!#6;!#1]')) else 0
        functional_groups.append([carboxylic_acid, amine, alcohol,ketone,ester,thiol,halogen_bond_acceptor,halogen_bond_donor])

    # Normalizing features
    scaler = StandardScaler()
    features = np.column_stack([mw, logp, tpsa, sc_count, rings, atom_types, ecfp6_fps, aromaticity, logd, degree_centrality, chirality, functional_groups])
    features = scaler.fit_transform(features)

    
    augment = sp.vstack(map(sp.csr_matrix, features))
    return sp.hstack([x, augment], format='csr')