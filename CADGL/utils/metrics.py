import torch
from operator import itemgetter
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from sklearn import metrics

from utils.utils import *

def cal_roc_auc_score(pred, label, edge_types):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    return roc_auc_score(all_label, all_pred)

def cal_acc_score(pred, label, edge_types):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    all_pred = torch.where(all_pred > 0.5, 1, 0)
    return accuracy_score(all_label, all_pred)

def cal_average_precision_score(pred, label):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    p, r, t = metrics.precision_recall_curve(all_label, all_pred)
    auc_prc = metrics.auc(r, p)
    return auc_prc


def cal_f1(pred, label):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    all_pred = torch.where(all_pred > 0.5, 1, 0)
    f1 = metrics.f1_score(all_label, all_pred)
    return f1

def cal_p(pred, label):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    all_pred = torch.where(all_pred > 0.5, 1, 0)
    p = metrics.precision_score(all_label, all_pred)
    return p

def cal_r(pred, label):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    all_pred = torch.where(all_pred > 0.5, 1, 0)
    r = metrics.recall_score(all_label, all_pred)
    return r