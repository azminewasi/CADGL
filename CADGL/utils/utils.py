import torch
from operator import itemgetter

def concat_all(item_dict):
    y = torch.cat(list(item_dict.values())).cpu()
    return y