import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T

from model_utils.graph_encoder.preprocessors.lcp import *
from model_utils.graph_encoder.preprocessors.mcp import *
from model_utils.graph_encoder.SSGAttn import *

    
class ContextAwareDeepGraphEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = LocalContextProcessor((-1, -1), hidden_channels)
        self.norm = pyg_nn.GraphNorm(hidden_channels)
        self.conv2 = MolecularContextProcessor((-1, -1), hidden_channels)
        self.conv3 = SSGAttn_Block(-1, out_channels,dropout=0.6, attention_type='MX',edge_sample_ratio=0.8, is_undirected=True) 

    def forward(self, x, edge_index):
        x1 = self.norm(self.conv1(x, edge_index)).relu()
        x2 = self.norm(self.conv2(x, edge_index)).relu()
        x = torch.cat((x1, x2))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        ss_loss = self.conv3.get_attention_loss()
        return x, ss_loss