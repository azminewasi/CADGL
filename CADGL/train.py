import os
import pandas as pd 
import json
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T
import time
from sklearn.model_selection import train_test_split
from tdc.chem_utils import featurize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from support import support_tasks
from data.data_utils import convert_to_hetero_graph, generate_label, count_num_labels
from model import CADGL_VGAE
from data.data_reader import read_files
from config.config import *
from utils.metrics import *


import warnings
warnings.filterwarnings("ignore")

train_df, valid_df, test_df, drug_2_smiles, drug_2_idx, relation_id = read_files()

train_data = convert_to_hetero_graph(train_df, drug_2_smiles, drug_2_idx, relation_id)

device = "cuda" if torch.cuda.is_available() else "cpu"

decoder_2_relation = {
    "mlp" : [edge_type[1] for edge_type in train_data.edge_types]
}
relation_2_decoder = {
    edge_type[1] : "mlp" for edge_type in train_data.edge_types
}

kl_lambda = {
    "drug" : 0.001,
    "gene" : 0.001,   
}

train_data = train_data.to(device)

train_edge_label_index_dict, train_edge_label_dict = generate_label(train_df, drug_2_idx, relation_id, device)
valid_edge_label_index_dict, valid_edge_label_dict = generate_label(valid_df, drug_2_idx, relation_id, device)
test_edge_label_index_dict, test_edge_label_dict = generate_label(test_df, drug_2_idx, relation_id, device)

model = CADGL_VGAE(128, 64, 64, train_data.metadata(), decoder_2_relation, relation_2_decoder,
                device).to(device)

optimizer = torch.optim.AdamW(list(model.parameters()),lr = 1e-3)



Epoch=[]
Loss=[]
Val_ROC=[]
Val_Acc=[]
Val_auprc=[]
f1=[]
p=[]
r=[]
Time=[]


for epoch in range(num_total_epoch):
    start = time.time()
    model.train()
    optimizer.zero_grad()
    latent_info_encoder,ss_loss = model.encode(train_data.x_dict, train_data.edge_index_dict)
    kl_loss = model.kl_loss_all(reduce = "ratio", lambda_ = kl_lambda)
    recon_loss = model.recon_loss_all_relation(latent_info_encoder, train_edge_label_index_dict)
    loss = recon_loss + kl_loss + ss_loss
    loss.backward()
    optimizer.step()
    loss = loss.detach().item()
  
    model.eval()
    with torch.no_grad():
        latent_info_encoder, _ = model.encode(train_data.x_dict, train_data.edge_index_dict)
        pos_edge_label_index_dict = valid_edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}
        for id_ in relation_id:
            relation = str(id_)
            src = "drug"
            dst = "drug"
            edge_type = (src, relation, dst)
            num_nodes = train_data.num_nodes
            neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[relation], num_nodes = num_nodes)
            edge_label_index_dict[relation] = torch.cat([pos_edge_label_index_dict[relation], 
                                                            neg_edge_label_index], dim = -1)

            pos_label = torch.ones(pos_edge_label_index_dict[relation].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
        
        edge_pred = model.decode_all_relation(latent_info_encoder, edge_label_index_dict) 
  
        for relation in edge_pred.keys():
            edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
        roc_auc = cal_roc_auc_score(edge_pred, edge_label_dict, relation_id)
        acc = cal_acc_score(edge_pred, edge_label_dict, relation_id)
        auprc = cal_average_precision_score(edge_pred, edge_label_dict)
        f1x =cal_f1(edge_pred, edge_label_dict)
        px=cal_p(edge_pred, edge_label_dict)
        rx=cal_r(edge_pred, edge_label_dict)

    end = time.time()
    print("- | Epoch: {:3d} | Loss: {:5.5f} | Val ROC: {:5.5f} | Val Acc: {:5.5f} | Val auprc: {:5.5f} | Val F1: {:5.5f} | Val P: {:5.5f} | Val R: {:5.5f} |Time: {:5.5f}".format(epoch+1,
                                                                                                        loss, 
                                                                                                        roc_auc,
                                                                                                        acc, 
                                                                                                        auprc,
                                                                                                        f1x,px,rx,
                                                                                                        end - start))
    Epoch.append(epoch+1)
    Loss.append(loss)
    Val_ROC.append(roc_auc)
    Val_Acc.append(acc)
    Val_auprc.append(auprc)
    f1.append(f1x)
    p.append(px)
    r.append(rx)
    Time.append(end - start)

    

model.eval()
with torch.no_grad():
    latent_info_encoder, _ = model.encode(train_data.x_dict, train_data.edge_index_dict)

    pos_edge_label_index_dict = test_edge_label_index_dict
    edge_label_index_dict = {}
    edge_label_dict = {}
    for id_ in relation_id:
        relation = str(id_)
        src = "drug"
        dst = "drug"
        edge_type = (src, relation, dst)
        num_nodes = train_data.num_nodes
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[relation], num_nodes = num_nodes)
        edge_label_index_dict[relation] = torch.cat([pos_edge_label_index_dict[relation], 
                                                        neg_edge_label_index], dim = -1)

        pos_label = torch.ones(pos_edge_label_index_dict[relation].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
    
    edge_pred = model.decode_all_relation(latent_info_encoder, edge_label_index_dict) 

    for relation in edge_pred.keys():
        edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
    roc_auc = cal_roc_auc_score(edge_pred, edge_label_dict, relation_id)
    acc = cal_acc_score(edge_pred, edge_label_dict, relation_id)
    auprc = cal_average_precision_score(edge_pred, edge_label_dict)
    f1x =cal_f1(edge_pred, edge_label_dict)
    px=cal_p(edge_pred, edge_label_dict)
    rx=cal_r(edge_pred, edge_label_dict)
    Epoch.append("Test")
    Loss.append("-")
    Val_ROC.append(roc_auc)
    Val_Acc.append(acc)
    Val_auprc.append(auprc)
    f1.append(f1x)
    p.append(px)
    r.append(rx)
    Time.append(end - start)

print('\n\nTest ROC: {:5.5f} | Test Acc: {:5.5f} | Test AUPRC: {:5.5f} | | Val F1: {:5.5f} | Val P: {:5.5f} | Val R: {:5.5f}'.format(roc_auc, acc, auprc,f1x,px,rx))
df=pd.DataFrame({"Epoch":Epoch,
                     "Loss":Loss,
                     "Val_ROC":Val_ROC,
                     "Val_Acc":Val_Acc,
                     "Val_auprc":Val_auprc,
    "F1":f1,
    "Precision":p,
    "Recall":r})
df.to_csv("train_data_scores.csv")