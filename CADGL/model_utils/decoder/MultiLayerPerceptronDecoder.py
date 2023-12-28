import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class MultiLayerPerceptronDecoder(nn.Module):
    def __init__(self, latent_dim, relations):
        super().__init__()
        
        self.mlps = nn.ModuleDict()

        for relation in relations:
            self.mlps[relation] = nn.Sequential(
                nn.Linear(latent_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )

    def forward(self, z, edge_index, relation):
        if isinstance(z, tuple):
            src = z[0][edge_index[0]]
            dst = z[1][edge_index[1]]
        else:
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
        out = torch.cat([src, dst], dim = 1)
        return self.mlps[relation](out)