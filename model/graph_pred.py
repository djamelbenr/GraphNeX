from __future__ import print_function
from datetime import date
import os
import sys
import argparse
import random
import pickle
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from utils import maskedMSE, maskedMSETest

from torch_geometric.data import Data, DataLoader

from modules.stp_base_model import STP_Base_Net
from modules.stp_r_model import STP_R_Net
from modules.stp_g_model import STP_G_Net #STP_G_Net
from modules.stp_gr_model import STP_GR_Net
#from modeles.stp_cgconv_nets import STP_CGConv_Net

#from mtp_gr_model import MTP_GR_Net

from stp_gr_dataset import STP_GR_Dataset
#from mtp_gr_dataset import MTP_GR_Dataset

import math
import time
from sklearn.model_selection import train_test_split

#models : 
#1. Basse Model: 
'''
The base model includes: LSTM , GAT, LSTM + FC
'''
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
#-- CRAT-PRED Model
#class crat_pred(torch.nn.Module):
#    def __init__(self, args):
#        super(crat_pred, self).__init__()
#        self.__annotations__
#        self.__annotations__
#        self.__annotations__
#        self.__dict__
#        ... #the rest of the code

#    return fut_pred



#0.--
class STP_Base_Model(torch.nn.Module):
    def __init__(self, args):
        super(STP_Base_Net, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        
    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[0],Hist_Enc.shape[1],-1))))
        return Hist_Enc
    
    def forward(self, data_pyg):
        hist = data_pyg.x
        fut = data_pyg.y
        hist_enc = self.LSTM_Encoder(hist)
        fut_pred = self.decode(hist_enc)
        return fut_pred
        
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
#1.--
class STP_Base_Net(torch.nn.Module):
    def __init__(self, args):
        super(STP_Base_Net, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
       
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
       
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc
    
    def forward(self, data_pyg):
        raiseNotImplementedError("forward is not implemented in STP_Base_Net!")
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
#2.--
class STP_G_Net(STP_Base_Net):
    def __init__(self, args):
        super(STP_G_Net, self).__init__(args)
        self.args = args
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(1*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
    
    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))
        return GAT_Enc
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)
        fut_pred = self.decode(fwd_tar_GAT_Enc)
        return fut_pred
    

#2.1-- Convoluration Graph Network
     
#3.--
class STP_R_Net(STP_Base_Net):
    def __init__(self, args):
        super(STP_R_Net, self).__init__(args)
        self.dec_rnn = torch.nn.LSTM(self.args['dyn_embedding_size'], self.args['decoder_size'], 2, batch_first=True)

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]
        fut_pred = self.decode(fwd_tar_LSTM_Enc)
        return fut_pred
#4.--
class STP_GR_Net(STP_G_Net):
    def __init__(self, args):
        super(STP_GR_Net, self).__init__(args)
        self.args = args
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n single TP or multiple TP? \n\n')
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]
        enc = torch.cat((fwd_tar_LSTM_Enc, fwd_tar_GAT_Enc), 1)
        fut_pred = self.decode(enc)
        return fut_pred
    
#5-
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

class STP_GR_Net_2(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]

        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc

    
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred   

#5.1 
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
#from torch_geometric.nn import CGConv
from torch_geometric.nn import conv
# seems working
class STP_GR_Net_31(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_31, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.cgc_conv1 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        self.cgc_conv2 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)

        self.nbrs_fc = torch.nn.Linear(self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc

    def CGCNN_Interaction(self, hist_enc, edge_index, edge_attr, target_index):
        node_matrix = hist_enc
        cgc_feature = self.cgc_conv1(node_matrix, edge_index, edge_attr)
        cgc_feature = self.cgc_conv2(cgc_feature, edge_index, edge_attr)
        target_cgc_feature = cgc_feature[target_index]

        CGCNN_Enc = self.leaky_relu(self.nbrs_fc(target_cgc_feature))

        return CGCNN_Enc

    
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)

        if data_pyg.edge_attr is not None:
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), data_pyg.edge_attr.float(), target_index)
        else:
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), None, target_index)

        #tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), data_pyg.edge_attr.float(), target_index)

        enc = torch.cat([Hist_Enc[target_index], tar_CGCNN_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred

#5.2-- ---


# import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
#from torch_geometric.nn import CGConv
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

class AgentGnn(nn.Module):
    def __init__(self, args):
        super(AgentGnn, self).__init__()
        self.args = args
        #self.latent_size = args.latent_size

        self.gcn1 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)

    def forward(self, gnn_in, centers, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
            agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        return gnn_out

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []
        offset = 0
        for i in range(len(agents_per_sample)):
            num_nodes = agents_per_sample[i]
            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)
            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]
            edge_index.append(edge_index_subgraph)
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        return edge_index

    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)
        rows, cols = edge_index
        edge_attr = data[cols] - data[rows]
        return edge_attr
    
###################################
#
###################################
class AgentGnn(nn.Module):
    def __init__(self, args):
        super(AgentGnn, self).__init__()
        self.args = args
        self.latent_size = args.latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

    def forward(self, gnn_in, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
            agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        return gnn_out

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):

            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        return edge_index

    def build_edge_attr(self, edge_index):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = rows - cols

        return edge_attr
    
##################################


    

#5.3---
class STP_GR_Net_5(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_5, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.cgc_conv1 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        self.cgc_conv2 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)

        self.nbrs_fc = torch.nn.Linear(self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc
    
    
    def define_edge_attr(self, data_pyg):
        # Define edge_attr using distance between nodes
        edge_index = data_pyg.edge_index
        node_pos = data_pyg.pos
        num_nodes = data_pyg.num_nodes
        edge_attr = torch.cdist(node_pos[:, edge_index[0]], node_pos[:, edge_index[1]], p=2.0)
        edge_attr = edge_attr.view(-1, 1).repeat(1, 2).view(-1)
        return edge_attr

    def CGCNN_Interaction(self, hist_enc, edge_index, edge_attr, target_index):
        node_matrix = hist_enc
        cgc_feature = self.cgc_conv1(node_matrix, edge_index, edge_attr)
        cgc_feature = self.cgc_conv2(cgc_feature, edge_index, edge_attr)
        target_cgc_feature = cgc_feature[target_index]

        CGCNN_Enc = self.leaky_relu(self.nbrs_fc(target_cgc_feature))

        return CGCNN_Enc

    
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        if data_pyg.edge_attr is not None:
            edge_attr = data_pyg.edge_attr.float()
        else:
            edge_attr = None
        #if data_pyg.edge_attr is not None:
        tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), edge_attr, target_index)

        dec_input = torch.cat([Hist_Enc[target_index], tar_CGCNN_Enc], dim=1)
        fut_pred = self.decode(dec_input)

        return fut_pred
    
#5.4--
from torch_geometric.nn import GCNConv

class STP_GR_Net_6(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_6, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gcn_conv1 = GCNConv(self.args['encoder_size'], self.args['encoder_size'])
        self.gcn_conv2 = GCNConv(int(self.args['encoder_size'])+self.args['encoder_size'], self.args['encoder_size'])
        self.nbrs_fc = torch.nn.Linear(int(self.args['encoder_size'])+self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc

    #def GCN_Interaction(self, hist_enc, edge_idx, target_index):
    #    node_matrix = hist_enc
    #    gcn_feature = self.gcn_conv1(node_matrix, edge_idx)
    #    gcn_feature = self.gcn_conv2(torch.cat([node_matrix, gcn_feature], dim=1), edge_idx)
    #    target_gcn_feature = gcn_feature[target_index]

    #    GCN_Enc = self.leaky_relu(self.nbrs_fc(target_gcn_feature))

    #    return GCN_Enc
    
    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gcn_conv1(node_matrix, edge_idx)
        gat_feature = self.gcn_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]

        # Flatten the target_gat_feature tensor before passing to linear layer
        target_gat_feature = target_gat_feature.view(target_gat_feature.size(0), -1)

        # Pass the flattened tensor through the linear layer
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc

    
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        tar_GCN_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        enc = torch.cat([Hist_Enc[target_index], tar_GCN_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred 
#6.--
class MultiheadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention, self).__init__()
        self.args = args
        self.latent_size = self.args.latent_size
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, 4)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            padded_att_in = torch.zeros(
                (len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)
            mask = torch.arange(max_agents) < torch.tensor(
                agents_per_sample)[:, None]
            padded_att_in[mask] = att_in
            mask_inverted = ~mask
            mask_inverted = mask_inverted.to(att_in.device)
            padded_att_in_swapped = torch.swapaxes(padded_att_in, 0, 1)
            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in_swapped, padded_att_in_swapped, padded_att_in_swapped, key_padding_mask=mask_inverted)
            padded_att_in_reswapped = torch.swapaxes(
                padded_att_in_swapped, 0, 1)
            att_out_batch = [x[0:agents_per_sample[i]]
                             for i, x in enumerate(padded_att_in_reswapped)]
        else:
            att_in = torch.split(att_in, agents_per_sample)
            for i, sample in enumerate(att_in):
                att_in_formatted = sample.unsqueeze(1)
                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted)
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)
        return att_out_batch            






############################################
#  END MODELS LOADER
############################################

class STP_GR_Net_4(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_4, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.cgc_conv1 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        self.cgc_conv2 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)

        self.nbrs_fc = torch.nn.Linear(self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc
    
    
    def define_edge_attr(data_pyg):
        # Define edge_attr using distance between nodes
        edge_index = data_pyg.edge_index
        node_pos = data_pyg.pos
        num_nodes = data_pyg.num_nodes
        edge_attr = torch.cdist(node_pos[:, edge_index[0]], node_pos[:, edge_index[1]], p=2.0)
        edge_attr = edge_attr.view(-1, 1).repeat(1, 2).view(-1)
        return edge_attr

    def CGCNN_Interaction(self, hist_enc, edge_index, edge_attr, target_index):
        node_matrix = hist_enc
        cgc_feature = self.cgc_conv1(node_matrix, edge_index, edge_attr)
        cgc_feature = self.cgc_conv2(cgc_feature, edge_index, edge_attr)
        target_cgc_feature = cgc_feature[target_index]

        CGCNN_Enc = self.leaky_relu(self.nbrs_fc(target_cgc_feature))

        return CGCNN_Enc

    
    def decode(self,enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        #enc = enc.permute(1,0,2) # permute
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)

        if data_pyg.edge_attr is not None:
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), data_pyg.edge_attr.float(), target_index)
        else:
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), None, target_index)

        fut_pred = self.decode(torch.cat([Hist_Enc.view(-1, 1, self.args['encoder_size']), tar_CGCNN_Enc.unsqueeze(1)], dim=2))
        print('fut_pred: ', np.shape(fut_pred))
        return fut_pred    
#############################################
# LOAD OTHER MODEL
#############################################
#-- data loader
class NGSIM_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path='xy/stp_data_2021', scenario_names=['stp0750am-0805am','stp0805am-0820am','stp0820am-0835am','stp0515-0530','stp0500-0515','stp0400-0415'], split='train', random_state=42):
        super(NGSIM_Dataset, self).__init__()
        
        self.data_path = data_path
        self.scenario_names = scenario_names
        self.split = split
        self.random_state = random_state
        assert self.split in ['train', 'val', 'test'], 'Invalid split name'
        self.all_data_names = os.listdir(self.data_path)
        self.scenario_data_names = [dn for dn in self.all_data_names if dn.split('_')[0] in self.scenario_names]
        if self.split == 'train':
            self.scenario_data_names, self.val_test_data_names = train_test_split(self.scenario_data_names, test_size=0.2, random_state=self.random_state)
            self.train_data_names, self.val_data_names = train_test_split(self.scenario_data_names, test_size=0.1, random_state=self.random_state)
        elif self.split == 'val':
            self.train_data_names, self.val_data_names = train_test_split(self.scenario_data_names, test_size=0.1, random_state=self.random_state)
        elif self.split == 'test':
            self.test_data_names, self.test_data_names = train_test_split(self.scenario_data_names, test_size=0.1, random_state=self.random_state)
            self.test_data_names = [dn for dn in self.all_data_names if dn not in self.scenario_data_names]

    def __len__(self):
        'Denotes the total number of samples'
        if self.split == 'train':
            return len(self.train_data_names)
        elif self.split == 'val':
            return len(self.val_data_names)
        else:
            return len(self.test_data_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.split == 'train':
            ID = self.train_data_names[index]
        elif self.split == 'val':
            ID = self.val_data_names[index]
        else:
            ID = self.test_data_names[index]
        data_item = torch.load(os.path.join(self.data_path, ID))
        data_item.x = data_item.node_feature.float()
        data_item.y = data_item.y.float()
        return data_item
    
def train_a_model(model_to_tr, num_ep=1):
    model_to_tr.train()

    print(optimizer.state_dict()['param_groups'][0]['lr'])

    train_running_loss = 0.0

    for i, data in enumerate(trainDataloader):
        # down-sampling data
        data.x =  data.x[:, ::2, :]
        data.y =  data.y[:,::2,:]  # have been changed to fit with the observation length applied during the learning process
        print('Observation dimension:', np.shape(data.x) )
        print('Future dimension:', np.shape(data.y) )
    
        optimizer.zero_grad()
        # forward + backward + optimize
        fut_pred = model_to_tr(data.to(args['device']))
        

        # init the mask 
        
        
        op_mask = torch.ones(data.y.shape)


        #print('shape mask: ', np.shape(op_mask))

        train_l = maskedMSE(fut_pred, data.y, op_mask)

        train_l.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 10)
        optimizer.step()
        train_running_loss += train_l.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('ep {}, {} batches, {} - {}'.format( num_ep, i + 1, 'maskedMSE', round(train_running_loss / 1000, 4)))
            train_running_loss = 0.0
    scheduler.step()
    return round(train_running_loss / (i%1000), 4)

def val_a_model(model_to_val):
    model_to_val.eval()
    #lossVals = torch.zeros(10)
    #counts = torch.zeros(10)

    lossVals = torch.zeros(25)
    counts = torch.zeros(25)
    
    with torch.no_grad():
        print('Testing no grad')
        # val_running_loss = 0.0
        for i, data in enumerate(valDataloader):
            # down-sampling data
            data.x =  data.x[:, ::2, :] 
            #data.y =  data.y[:,4::5,:] 
            data.y =  data.y[:,::2,:] 

            # predict
            fut_pred = model_to_val(data.to(args['device']))
            # calculate loss
            fut_pred = fut_pred.permute(1,0,2)
            ff = data.y.permute(1,0,2)
            #print('ff_shape: ', ff.shape)
            op_mask = torch.ones(ff.shape)
            print('mask_dim: ', np.shape(op_mask))
            l, c = maskedMSETest(fut_pred, ff, op_mask)
            
            #--Detach
            lossVals +=l.detach()
            counts += c.detach()
    rmseOverall=torch.pow(lossVals / counts,0.5) *0.3048
    # print the prediction outcome :: TODO:: keep
    print("Prediction RMSE: ", torch.pow(lossVals / counts,0.5) *0.3048)   
    print("Prediction RMSE over 5s: ", rmseOverall[4::5])
    print("RMSE (m):", rmseOverall[4::5],  "Mean= : ", np.array(rmseOverall[4::5]).mean())
    #print("Prediction NLL: ", )
    return torch.pow(lossVals / counts,0.5) *0.3048

def test_a_model(model_to_test):
    model_to_test.eval()
    #lossVals = torch.zeros(10)
    #counts = torch.zeros(10)

    lossVals = torch.zeros(25)
    counts = torch.zeros(25)
    
    with torch.no_grad():
        print('Testing no grad')
        # val_running_loss = 0.0
        for i, data in enumerate(testDataloader):
            # down-sampling data
            data.x =  data.x[:, ::2, :] 
            #data.y =  data.y[:,4::5,:] 
            data.y =  data.y[:,::2,:] 

            # predict
            fut_pred = model_to_test(data.to(args['device']))

            # calculate loss
            fut_pred = fut_pred.permute(1,0,2)
            ff = data.y.permute(1,0,2)
            
            #print('ff_shape: ', ff.shape)
            
            op_mask = torch.ones(ff.shape)
            print('mask_dim: ', np.shape(op_mask))

            ll, cc = maskedMSETest(fut_pred, ff, op_mask) 
            #--Detach
            lossVals +=ll.detach()
            counts += cc.detach()
    rmseOverall=torch.pow(lossVals / counts,0.5) *0.3048
    # print the prediction outcome :: TODO:: keep
    print("Prediction RMSE: ", torch.pow(lossVals / counts,0.5) *0.3048)   
    print("Prediction RMSE over 5s: ", rmseOverall[4::5])
    print("RMSE (m):", rmseOverall[4::5],  "Mean= : ", np.array(rmseOverall[4::5]).mean())
    #print("Prediction NLL: ", )
    return torch.pow(lossVals / counts,0.5) *0.3048


def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    
    # can be modified with a parent parser
    # command line arguments
    parent_parser = pprint.PrettyPrinter(indent=1)
    def parse_args(cmd_args):
        """ Parse arguments from command line input
        """
        parser = argparse.ArgumentParser(description='Training parameters')
        parser.add_argument('-graph_encoder', '--gnn', type=str, default='GAT', help="the GNN to be used")
        parser.add_argument('-RNN_encoder', '--rnn', type=str, default='GRU', help="the RNN to be used")
        parser.add_argument('-model', '--modeltype', type=str, default='GR', help="the model type [R, G, GR]")
        parser.add_argument('-observation', '--histlength', type=int, default=16, help="length of history 10, 30, 50") #16 is 3sec
        parser.add_argument('-prediction', '--futlength', type=int, default=25, help="length of future 50") # 25 is 5sec
        parser.add_argument('-gpus', '--gpu', type=str, default='0', help="the GPU to be used")
        parser.add_argument('-i', '--number', type=int, default=0, help="run times of the py script")
        parser.add_argument('-epochs', '--epoch', type=int, default=50, help="number of training epochs")
        # model settings
        parser.add_argument('-latent_size', '--latent_size', type=int, default=128, help="latent space dimension")
        parser.set_defaults(render=False)
        return parser.parse_args(cmd_args)

    # Parse arguments
    cmd_args = sys.argv[1:]
    cmd_args = parse_args(cmd_args)

    ## Network Arguments
    args = {}
    args['run_i'] = cmd_args.number
    args['random_seed'] = 1
    args['input_embedding_size'] = 16 # if args['single_or_multiple'] == 'single_tp' else 32
    args['encoder_size'] = 128#32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['decoder_size'] = 64#64 #default = 64 # if args['single_or_multiple'] == 'single_tp' else 128 # 128 256
    args['dyn_embedding_size'] = 32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128

    args['edge_filters']=32

    args['train_epoches'] = 1#50
    args['num_gat_heads'] = 8 #3 #default=3
    args['concat_heads'] = True # False # True
    
    args['in_length'] = cmd_args.histlength
    args['out_length'] = cmd_args.futlength
    
    args['single_or_multiple'] = 'single_tp' # or multiple_tp single_tp
    args['date'] = date.today().strftime("%b-%d-%Y")
    args['batch_size'] = 64 if args['single_or_multiple'] == 'single_tp' else 128 # default is 16

    #args['batch_size'] = 16 if args['single_or_multiple'] == 'single_tp' else 128
    args['net_type'] = cmd_args.modeltype
    args['enc_rnn_type'] = cmd_args.rnn # LSTM GRU
    args['gnn_type'] = cmd_args.gnn # GCN GAT
    
    device = torch.device('cuda:{}'.format(cmd_args.gpu) if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # set random seeds
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])

    if device != 'cpu':
        print('running on {}'.format(device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        print('seed setted! {}'.format(args['random_seed']))
    

    # Initialize network
    if args['net_type'] == 'GR':
        if args['single_or_multiple'] == 'single_tp':
            print('loading {} model'.format(args['net_type']))
            train_net = STP_GR_Net(args)
    elif args['net_type'] == 'R':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_R_Net(args)
    elif args['net_type'] == 'G':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_G_Net(args)
    else:
        print('\nselect a proper model type!\n')


    train_net = STP_GR_Net_4(args)
    #train_net = STP_GR_Net_2(args)
    train_net.to(args['device'])

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))
    print('NET: ', train_net)

    parent_parser.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))
   
    optimizer = torch.optim.Adam(train_net.parameters(),lr=0.001) 
    scheduler = MultiStepLR(optimizer, milestones=[1], gamma=1.0)

    if args['single_or_multiple'] == 'multiple':
        optimizer = torch.optim.Adam(train_net.parameters(),lr=0.004) # lr 0.0035, batch_size=4 or 8.
        scheduler = MultiStepLR(optimizer, milestones=[1,2,3,6,20,30], gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6], gamma=1.0)
    

    # NGSIM US101 and i80
    if args['single_or_multiple'] == 'single_tp':
        train_set = NGSIM_Dataset(data_path='./xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am','stp0820am-0835am', 
                                                                                    'stp0515-0530','stp0500-0515','stp0400-0415'], split='train') 
        val_set = NGSIM_Dataset(data_path='./xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am','stp0820am-0835am', 
                                                                                    'stp0515-0530','stp0500-0515','stp0400-0415'], split='val') 
        
        # Test set
        test_set=NGSIM_Dataset(data_path='./xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am','stp0820am-0835am', 
                                                                                    'stp0515-0530','stp0500-0515','stp0400-0415'], split='test') 
    elif args['single_or_multiple'] == 'multiple_tp':
        #train_set = STP_GR_Dataset(data_path='./xy/gat_mtp_data_0805am_Train/') # HIST_1w FUT_1w HIST FUT
        #val_set = STP_GR_Dataset(data_path='./xy/gat_mtp_data_0805am_Test/')
        train_set = NGSIM_Dataset(data_path='./xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am', 'stp0820am-0835am',
                                                                                   'stp0515-0530', 'stp0500-0515','stp0400-0415'], split='train') # HIST_1w FUT_1w HIST FUT

        val_set = NGSIM_Dataset(data_path='./xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am', 'stp0820am-0835am',
                                                                                 'stp0515-0530','stp0500-0515','stp0400-0415'], split='val')
    # load training and validation set
    torch.set_num_threads(4)
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    valDataloader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []

    min_val_loss = 1000.0

    for ep in range(1, args['train_epoches']+1):
        train_loss_ep = train_a_model(train_net, num_ep=ep)
        val_loss_ep = val_a_model(train_net)

        Val_LOSS.append(val_loss_ep)
        Train_LOSS.append(train_loss_ep)

        ## save model
        if val_loss_ep[-1]<min_val_loss:
            save_model_to_PATH = './trained_models/{}_{}_{}_{}_h{}f{}_d{}_{}.tar'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                                 args['in_length'], args['out_length'], '3s', args['batch_size'])
            torch.save(train_net.state_dict(), save_model_to_PATH)
            min_val_loss = val_loss_ep[-1]

        with open('./trained_models/{}-{}-{}-{}-h{}f{}-TRAINloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                            args['in_length'], args['out_length'], '3s', args['batch_size']), "w") as file:
            file.write(str(Train_LOSS))
        with open('./trained_models/{}-{}-{}-{}-h{}f{}-VALloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                   args['in_length'], args['out_length'], '3s', args['batch_size']), "w") as file:
            file.write(str(Val_LOSS))
        save_obj_pkl(args, save_model_to_PATH.split('.tar')[0])
    
    torch.save(train_net.state_dict(), save_model_to_PATH)

    '''
    
To write a test script, we need a specific problem to solve, and the code given does not define any problem statement. However, we can write a test script to ensure the required packages are installed and import correctly.

Here is an example of a test script that imports the required packages, loads the data, creates a dataloader, initializes the models, and trains the model:

python
Copy code
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader

from stp_gr_dataset import STP_GR_Dataset
from modules.stp_gr_model import STP_GR_Net
from utils import maskedMSE, maskedMSETest

# define constants
NUM_EPOCHS = 2
BATCH_SIZE = 16

# load data
train_data = STP_GR_Dataset(root='./data', split='train')
val_data = STP_GR_Dataset(root='./data', split='val')

# create data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# initialize the model
model = STP_GR_Net()

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = maskedMSE(out, batch.y, batch.op_mask)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_loss = []
        for batch in val_loader:
            out = model(batch)
            loss = maskedMSE(out, batch.y, batch.op_mask)
            val_loss.append(loss.item())
        print('Epoch:', epoch+1, 'Validation Loss:', np.mean(val_loss))
    '''
