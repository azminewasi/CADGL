import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T

from model_utils.graph_encoder.ContextAwareDeepGraphEncoder import ContextAwareDeepGraphEncoder
from model_utils.decoder.MultiLayerPerceptronDecoder import MultiLayerPerceptronDecoder

class CADGL_VGAE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, latent_dim, metadata, 
                        decoder_2_relation, relation_2_decoder, device, heads=1):
        super().__init__()

        self.node_type = metadata[0]
        self.hetero_model = ContextAwareDeepGraphEncoder(hidden_channels=hidden_channels, out_channels=out_channels)
        self.hetero_model = pyg_nn.to_hetero(self.hetero_model, metadata, aggr='mean')

        self.mu_combined_hidden_features_extractor = nn.ModuleDict()
        self.logstd_combined_hidden_features_extractor = nn.ModuleDict()
        for node_type in self.node_type:
        
          self.mu_combined_hidden_features_extractor[node_type] = nn.Sequential(
        nn.Linear(out_channels, latent_dim * 2),
        nn.LayerNorm(latent_dim * 2),
        nn.SELU(),
        nn.Linear(latent_dim * 2, 256),
        nn.SELU(),
        nn.Linear(256, latent_dim)
    )

          self.logstd_combined_hidden_features_extractor[node_type] = nn.Sequential(
        nn.Linear(out_channels, latent_dim * 2),
        nn.LayerNorm(latent_dim * 2),
        nn.SELU(),
        nn.Linear(latent_dim * 2, 256),
        nn.SELU(),
        nn.Linear(256, latent_dim)
    )

            

        self.__mu__ = {}
        self.__logstd__ = {}


        self.decoder = nn.ModuleDict()
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder
        self.decoder["mlp"] = MultiLayerPerceptronDecoder(latent_dim, self.decoder_2_relation["mlp"])

    def reparameterize(self, mu, logstd, ss_loss):
        latent_info_encoder = {}
        for node_type in self.node_type:
            latent_info_encoder[node_type] = mu[node_type] + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
        return latent_info_encoder, ss_loss

    def encode(self, x_dict, edge_index_dict):
        latent_info_encoder, ss_loss = self.hetero_model(x_dict, edge_index_dict)
        
        for node_type in self.node_type:
            self.__mu__[node_type] = self.mu_combined_hidden_features_extractor[node_type](latent_info_encoder[node_type])
            self.__logstd__[node_type] = self.logstd_combined_hidden_features_extractor[node_type](latent_info_encoder[node_type])


        for node_type in self.node_type:
            self.__logstd__[node_type] = self.__logstd__[node_type].clamp(max = 10)

        return self.reparameterize(self.__mu__, self.__logstd__,ss_loss)

    def decode(self, latent_info_encoder, edge_index, edge_type = None, sigmoid = False):
        src, relation, dst = edge_type
        decoder_type = self.relation_2_decoder[relation]
        z = (latent_info_encoder[src], latent_info_encoder[dst])
        output = self.decoder["mlp"](z, edge_index, relation)
        if sigmoid:
            output = F.sigmoid(output)
        return output

    def decode_all_relation(self, latent_info_encoder, edge_index_dict, sigmoid = False):
        output = {}
        for edge_type in edge_index_dict.keys():
            src, relation, dst = "drug", edge_type, "drug"
            decoder_type = self.relation_2_decoder[relation]
            z = (latent_info_encoder[src], latent_info_encoder[dst])
            output[relation] = self.decoder["mlp"](z, edge_index_dict[edge_type], relation)
            if sigmoid:
                output[relation] = F.sigmoid(output[relation])
        return output

    def kl_loss(self, node_type, mu = None, logstd = None):
        mu = self.__mu__[node_type] if mu is None else mu
        logstd = self.__logstd__[node_type] if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
                torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim = 1))

    def kl_loss_all(self, reduce = "sum", lambda_ = None):
        assert len(self.__mu__) > 0 and len(self.__logstd__) > 0
        loss = {}
        for node_type in self.node_type:
            loss[node_type] = self.kl_loss(node_type, self.__mu__[node_type], self.__logstd__[node_type])
        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "ratio":
            assert lambda_ is not None
            loss = sum([loss[node_type] * lambda_[node_type] for node_type in self.node_type])
        return loss

    def recon_loss(self, latent_info_encoder, edge_type, pos_edge_index, neg_edge_index = None):
        poss_loss = -torch.log(
                self.decode(latent_info_encoder, pos_edge_index, edge_type, sigmoid= True) + 1e-15).mean()

        if neg_edge_index is None:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = latent_info_encoder[src].shape[0], latent_info_encoder[dst].shape[0]
            neg_edge_index = pyg_utils.negative_sampling(pos_edge_index, num_nodes = (num_src_node, num_dst_node))

        neg_loss = -torch.log(1 -
                self.decode(latent_info_encoder, neg_edge_index, edge_type, sigmoid=True) + 1e-15).mean()

        return poss_loss + neg_loss

    def recon_loss_all_relation(self, latent_info_encoder, edge_index_dict, reduce = "sum"):
        loss = {}
        for relation in edge_index_dict.keys():
            edge_type = "drug", relation, "drug"
            loss[edge_type] = self.recon_loss(latent_info_encoder, edge_type, edge_index_dict[relation])

        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "mean":
            loss = sum(loss.values()) / len(edge_index_dict.keys())
        return loss