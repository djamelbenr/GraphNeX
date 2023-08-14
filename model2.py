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

# loading models --
from modules.stp_base_model import STP_Base_Net
from modules.stp_r_model import STP_R_Net
from modules.stp_g_model import STP_G_Net #STP_G_Net
from modules.stp_gr_model import STP_GR_Net
#from modeles.stp_cgconv_nets import STP_CGConv_Net

#from mtp_gr_model import MTP_GR_Net
# loading data processor --

from stp_gr_dataset import STP_GR_Dataset
#from mtp_gr_dataset import MTP_GR_Dataset
from scipy import sparse
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
# Residual Layer + Freezing ...
class DecoderResidual():
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output=[]
        for i in range(sum(args, mod_steps)):
            #mod_steps::prediction model TODO --single mode or multi-mode
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output)

    def forward(self, decoder_in, is_frozen):
        sample_wise_out=[]
        
        # TODO: testing, training with freezing , or training
        if self.training is False:
            for out_subnet in self.output: 
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:## mod_steps = 30 --- defined in the parser_model.add_argument
            for i in range(self.args.mod_steps[0],sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:
            sample_wise_out.append(self.append(self, output[0](decoder_in)))
        # concatenated the output 
        decoder_out=torch.stack(sample_wise_out)
        decoder_out=torch.swapaxes(decoder_out,0,1)
        return decoder_out
    
    def unfreeze_layers(self):
        for layer in range(self, args.mod_steps[0],sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad=True
#5-
class STP_Net2(torch.nn.Module):
    def __init__(self, args):
        super(STP_Net2, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
       
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
       
        self.dec_rnn = torch.nn.LSTM(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        
        if self.args['model_type'] == 'gat':
            self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.gat_conv2 = GATConv(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'], 1 * self.args['encoder_size'])
    
    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc
    
    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))
        return GAT_Enc
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        
        if self.args['model_type'] == 'gat':
            fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)
            fut_pred = self.decode(fwd_tar_GAT_Enc)
        else:
            fut_pred = self.decode(fwd_Hist_Enc)
        
        return fut_pred
    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    
#####################################
# A sophisticated way to rewrite the model 
####################################
class DualChannelModel(torch.nn.Module):
    def __init__(self, args):
        super(DualChannelModel, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        
        if args['model_type'] == 'base':
            self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
            self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        elif args['model_type'] == 'gat':
            self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
            self.dec_rnn = torch.nn.LSTM(1*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
            self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        elif args['model_type'] == 'r':
            self.dec_rnn = torch.nn.LSTM(self.args['dyn_embedding_size'], self.args['decoder_size'], 2, batch_first=True)
            self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        elif args['model_type'] == 'gr':
            self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
            self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
            self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
            self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        
    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[0], Hist_Enc.shape[1], -1))))
        return Hist_Enc
    
    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))
        return GAT_Enc
    
    def forward(self, data_pyg):
        hist = data_pyg.x
        fut = data_pyg.y
        hist_enc = self.LSTM_Encoder(hist)
        
        if self.args['model_type'] == 'base':
            fut_pred = self.decode(hist_enc)
        elif self.args['model_type'] == 'gat' or self.args['model_type'] == 'gr':
            if self.args['single_or_multiple'] == 'single_tp':
                target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
                target_index = torch.cat(target_index, dim=0)
            else:
                print('\n\nOnly single_tp is supported in R model?\n\n')
                return None
            
            if self.args['model_type'] == 'gat':
                fwd_tar_GAT_Enc = self.GAT_Interaction(hist_enc, data_pyg.edge_index.long(), target_index)
                enc = fwd_tar_GAT_Enc
            elif self.args['model_type'] == 'gr':
                fwd_tar_GAT_Enc = self.GAT_Interaction(hist_enc, data_pyg.edge_index.long(), target_index)
                fwd_tar_LSTM_Enc = hist_enc[target_index]
                enc = torch.cat((fwd_tar_LSTM_Enc, fwd_tar_GAT_Enc), 1)
            
            fut_pred = self.decode(enc)
        elif self.args['model_type'] == 'r':
            if self.args['single_or_multiple'] == 'single_tp':
                target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
                target_index = torch.cat(target_index, dim=0)
            else:
                print('\n\nOnly single_tp is supported in R model?\n\n')
                return None
            
            fwd_tar_LSTM_Enc = hist_enc[target_index]
            fut_pred = self.decode(fwd_tar_LSTM_Enc)
        
        return fut_pred
    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred

#####################################
# AUX modules 
#####################################
# Hardcode of the GATv2...
######################
# Citation: How Attentive are Graph Attention Networks?
######################
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


################################
#  GATv2 
################################

class GAT2Conv(MessagePassing):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the layer will share weights.
        (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GAT2Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=bias)
            self.lin_r = Linear(in_channels[1], heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


###################################
# Standard GAT implementation
###################################

class STP_GR_Net_4_copy(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_4_copy, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.cgc_conv1 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        print('cgc_conv1: ', self.cgc_conv1.size())

        self.cgc_conv2 = conv.CGConv(self.args['encoder_size'], dim=2, batch_norm=True)
        print('cgc_conv2: ', self.cgc_conv2.size())

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






####################################
#
####################################
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        h_prime_cat = torch.zeros(size=(input.shape[0],
                                        input.shape[2],
                                        input.shape[3],
                                        self.out_features)).to(input.device)

        for step_i in range(input.shape[2]):
            input_i = input[:, :, step_i, :]
            input_i = input_i.permute(0, 2, 1)
            adj_i = adj[:, step_i, :, :]
            Wh = torch.matmul(input_i, self.W)

            batch_size = Wh.size()[0]
            N = Wh.size()[1]  # number of nodes
            Wh_chunks = Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features)
            Wh_alternating = Wh.repeat(1, N, 1)
            combination_matrix = torch.cat([Wh_chunks, Wh_alternating], dim=2)
            a_input = combination_matrix.view(batch_size, N, N, 2 * self.out_features)

            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(adj_i > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, 0.25, training=self.training)
            h_prime = torch.matmul(attention, Wh)  # [8, 120, 64]
            h_prime_cat[:, step_i, :, :] = h_prime

        if self.concat:
            return F.elu(h_prime_cat)
            # return h_prime_return
        else:
            return h_prime_cat


class GATBlock(nn.Module):
    def __init__(self, input_dim, out_channels, stride=1, residual=True):
        super(GATBlock, self).__init__()

        self.att_1 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_2 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_out = GraphAttentionLayer(out_channels, out_channels, concat=False)

        if not residual:
            self.residual = lambda x: 0
        elif (input_dim == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels), )

    def forward(self, x, adjacency):
        res = self.residual(x)
        x_1 = self.att_1(x, adjacency)
        x_2 = self.att_2(x, adjacency)
        x = torch.stack([x_1, x_2], dim=-1).mean(-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.dropout(x, 0.25)
        x = F.elu(self.att_out(x, adjacency))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + res
        return x
    
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

class STP_GR_Net_21(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_21, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)

        self.gat_conv2 = GATConv(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'],self.args['encoder_size'],heads=self.args['num_gat_heads'],concat=self.args['concat_heads'],dropout=0.0)
        
        #self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
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

    ##################
    # add a residual decoder
    #def decode(self, enc):
    #    enc = enc.unsqueeze(1)
    #    enc = enc.repeat(1, self.args['out_length'], 1)
    #    h_dec, _ = self.dec_rnn(enc)
    #    fut_pred = self.op(h_dec)

    #    residual = enc[:, :, -self.args['encoder_size']:]
    #    residual = residual.view(residual.size(0), residual.size(1), residual.size(2), 1)
    #    residual = residual.repeat(1, 1, 1, fut_pred.size(3))

    #    fut_pred = fut_pred + residual
    #    return fut_pred
    
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


#######################################################
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





#-------------
class STP_GR_Net_2Bis(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2Bis, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)

        self.gat_conv2 = GATConv(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'],self.args['encoder_size'],heads=self.args['num_gat_heads'],concat=self.args['concat_heads'],dropout=0.0)
        
        self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2])))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]

        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
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

# GATv2 -- works
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

class STP_GR_Net_2GATv2BIS(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2GATv2BIS, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATv2Conv(
            self.args['encoder_size'], self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        
        self.gat_conv2 = GATv2Conv(
            self.args['num_gat_heads'] * self.args['encoder_size'], self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        
        self.nbrs_fc = nn.Linear(
            self.args['num_gat_heads'] * self.args['encoder_size'] + self.args['encoder_size'], self.args['encoder_size'])
        self.dec_rnn = nn.LSTM(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(Hist_Enc.squeeze(0)))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)

        target_gat_feature = gat_feature[target_index]
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc
    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)

        residual = enc[:, :, -self.args['encoder_size']:]
        residual = residual.view(residual.size(0), residual.size(1), residual.size(2), 1)
        residual = residual.repeat(1, 1, 1, fut_pred.size(3))

        fut_pred = fut_pred + residual
        return fut_pred

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        print('Hist Encod:', Hist_Enc.shape)

        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred

class STP_GR_Net_2GATv2BIS(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2GATv2BIS, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATv2Conv(
            self.args['encoder_size'], self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        
        self.gat_conv2 = GATv2Conv(
            (self.args['num_gat_heads'] - 1) * self.args['encoder_size'], 
            self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        
        self.nbrs_fc = nn.Linear(
            self.args['num_gat_heads'] * self.args['encoder_size'] + self.args['encoder_size'], self.args['encoder_size'])
        self.dec_rnn = nn.LSTM(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(Hist_Enc.squeeze(0)))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        
        batch_size = gat_feature.size(0)
        num_nodes = gat_feature.size(1)
        gat_feature = gat_feature.view(batch_size * num_nodes, -1)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        gat_feature = gat_feature.view(batch_size, num_nodes, -1)

        target_gat_feature = gat_feature[target_index]
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc
    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)

        residual = enc[:, :, -self.args['encoder_size']:]
        residual = residual.view(residual.size(0), residual.size(1), residual.size(2), 1)
        residual = residual.repeat(1, 1, 1, fut_pred.size(3))

        fut_pred = fut_pred + residual
        return fut_pred

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        print('Hist Encod: ', Hist_Enc.shape)

        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected
class STP_GR_Net_2GATv3(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2GATv3, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
       
        self.gat_conv1 = GATv2Conv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATv2Conv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        
        self.nbrs_fc = nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        
        self.dec_rnn = nn.LSTM(self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        ##self.dec_rnn = nn.LSTM(2 * self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = nn.Linear(2*self.args['encoder_size'], 2)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(Hist_Enc))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        print("gat_feature shape:", gat_feature.shape)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]
        print("gat_feature shape after conv2:", gat_feature.shape)
        #target_gat_feature = gat_feature.view(-1, gat_feature.size(1))
        print("target_gat_feature shape:", target_gat_feature.shape)
        
        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc

    def decode2(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)

        # Adjust the dimensions of the residual connection
        residual = enc[:, :, -self.args['encoder_size']:]
        residual = residual.view(residual.size(0), residual.size(1), 1, residual.size(2))
        residual = residual.repeat(1, 1, fut_pred.size(2), 1)

        # Add a residual connection
        fut_pred = fut_pred + residual

        # Reshape the fut_pred tensor
        fut_pred = fut_pred.view(fut_pred.size(0) * fut_pred.size(1), -1)

        return fut_pred

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)

        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)

        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred
    

class MultiheadSelfAttention_crat(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention_crat, self).__init__()
        self.args = args

        self.latent_size = 64
        self.num_heads = 4
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            
            print('max_agents: ', max_agents)    

            padded_att_in = torch.zeros(
                (len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)

            print('padded_att_in: ', padded_att_in)
            print('faulty tensor: ',  torch.tensor(agents_per_sample)[:, None])
            print('torch.arange(max_agents): ', torch.arange(max_agents))
            mask = torch.arange(max_agents, device=att_in[0].device) < torch.tensor(agents_per_sample)[:, None].to(device)
            print('shape attention in: ', np.shape(att_in))
            print('mask shape: ', np.shape(mask))
            print('padded_att_in shape: ', np.shape(padded_att_in))

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
            print('shape attention inputs: ', np.shape(att_in))
            for i, sample in enumerate(att_in):
                # Add the batch dimension (this has to be the second dimension, because attention requires it)
                att_in_formatted = sample.unsqueeze(1)

                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted)

                # Remove the "1" batch dimension
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)

        return att_out_batch


class STP_GR_Net_2GATv2_attention(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_2GATv2_attention, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        self.gat_conv1 = GATv2Conv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATv2Conv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.att_module = MultiheadSelfAttention(args)  # Added multi-head attention module
        self.dec_rnn = nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.op = nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc

    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        target_gat_feature = gat_feature[target_index]

        GAT_Enc = self.att_module(target_gat_feature, [target_gat_feature.shape[0]])

        return GAT_Enc

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred

latent_size = 64
num_heads   = 4
class MultiheadSelfAttention(nn.Module):
    def __init__(self, latent_size=64, num_heads=4):
        super(MultiheadSelfAttention, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            device = att_in[0].device

            padded_att_in = torch.zeros(
                len(agents_per_sample), max_agents, self.latent_size, device=device
            )

            for i, sample in enumerate(att_in):
                agents = agents_per_sample[i]
                if agents > 0:
                    sample = sample.unsqueeze(0)  # Add extra dimension
                    repeated_sample = sample.repeat(max_agents // agents,  1)
                    padded_att_in[i, :agents] = repeated_sample[:agents]  # Assign values

                    print("shape padded_att_in: ", np.shape(padded_att_in[i, :agents]))
                    print("shape repeated_sample[:agents]: ", np.shape(repeated_sample[:agents]))
                    
            mask = torch.arange(max_agents, device=device) < torch.tensor(
                agents_per_sample, device=device
            )[:, None]

            padded_att_in = padded_att_in.permute(1, 0, 2)
            mask_inverted = ~mask
            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in, padded_att_in, padded_att_in, key_padding_mask=mask_inverted
            )
            padded_att_in_reswapped = padded_att_in_swapped.permute(1, 0, 2)

            for i in range(len(agents_per_sample)):
                agents = agents_per_sample[i]
                if agents > 0:
                    att_out_batch.append(padded_att_in_reswapped[i, :agents])

        else:
            att_in = torch.split(att_in, agents_per_sample)
            for sample in att_in:
                if sample.size(1) > 0:
                    att_out, _ = self.multihead_attention(
                        sample, sample, sample
                    )
                    att_out_batch.append(att_out.squeeze(dim=1))

        return att_out_batch
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
'''
   args = {}
    args['run_i'] = cmd_args.number
    args['random_seed'] = 1
    args['input_embedding_size'] = 16 # if args['single_or_multiple'] == 'single_tp' else 32
    args['encoder_size'] = 64#32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['decoder_size'] = 64#64 #default = 64 # if args['single_or_multiple'] == 'single_tp' else 128 # 128 256
    args['dyn_embedding_size'] = 64 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['latent_size'] = 128

    args['edge_filters']=64

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
'''
def calc_prediction_metrics(preds, gts):
    # Calculate prediction error for each mode
    # Output has shape (batch_size, n_modes, n_timesteps)
    error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)

    # Calculate the error for the first mode (at index 0)
    fde_1 = np.average(error_per_t[:, 0, -1])
    ade_1 = np.average(error_per_t[:, 0, :])

    # Calculate the error for all modes
    # Best mode is always the one with the lowest final displacement
    lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
    error_per_t = error_per_t[np.arange(
        preds.shape[0]), lowest_final_error_indices]
    fde = np.average(error_per_t[:, -1])
    ade = np.average(error_per_t[:, :])
    return ade_1, fde_1, ade, fde

class MultiheadSelfAttention_crat(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention_crat, self).__init__()
        self.args = args

        self.latent_size = 64
        self.num_heads = 4
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            
            print('max_agents: ', max_agents)    

            padded_att_in = torch.zeros(
                (len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)

            print('padded_att_in: ', padded_att_in)
            print('faulty tensor: ',  torch.tensor(agents_per_sample)[:, None])
            print('torch.arange(max_agents): ', torch.arange(max_agents))
            mask = torch.arange(max_agents, device=att_in[0].device) < torch.tensor(agents_per_sample)[:, None].to(device)
            print('shape attention in: ', np.shape(att_in))
            print('mask shape: ', np.shape(mask))
            print('padded_att_in shape: ', np.shape(padded_att_in))

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
            print('shape attention inputs: ', np.shape(att_in))
            for i, sample in enumerate(att_in):
                # Add the batch dimension (this has to be the second dimension, because attention requires it)
                att_in_formatted = sample.unsqueeze(1)

                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted)

                # Remove the "1" batch dimension
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)

        return att_out_batch
'''
class MultiheadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention, self).__init__()
        self.latent_size = 64
        self.num_heads = 4
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            device = att_in[0].device

            padded_att_in = torch.zeros(
                len(agents_per_sample), max_agents, self.latent_size, device=device
            )

            for i, sample in enumerate(att_in):
                agents = agents_per_sample[i]
                padded_att_in[i, :agents] = sample[:agents]

            mask = torch.arange(max_agents, device=device) < torch.tensor(
                agents_per_sample, device=device
            )[:, None]

            padded_att_in = padded_att_in.permute(1, 0, 2)
            mask_inverted = ~mask
            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in, padded_att_in, padded_att_in, key_padding_mask=mask_inverted
            )
            padded_att_in_reswapped = padded_att_in_swapped.permute(1, 0, 2)

            for i in range(len(agents_per_sample)):
                agents = agents_per_sample[i]
                att_out_batch.append(padded_att_in_reswapped[i, :agents])

        else:
            att_in = torch.split(att_in, agents_per_sample)
            for sample in att_in:
                att_out, _ = self.multihead_attention(
                    sample, sample, sample
                )
                att_out_batch.append(att_out.squeeze(dim=1))

        return att_out_batch
    
class STP_GR_Net_Attention(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_Attention, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])

        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # Attention module:
        self.attention = MultiheadSelfAttention(self.args['encoder_size'])  # Added MultiheadSelfAttention

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

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
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
        Hist_Enc = self.attention(Hist_Enc, data_pyg.batch) # Apply attention to LSTM output
        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred
'''    
latent_size = 64
num_heads   = 2
class MultiheadSelfAttention(nn.Module):
    def __init__(self, latent_size=64, num_heads=2):
        super(MultiheadSelfAttention, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            device = att_in[0].device

            padded_att_in = torch.zeros(
                len(agents_per_sample), max_agents, self.latent_size, device=device
            )

            for i, sample in enumerate(att_in):
                agents = agents_per_sample[i]
                if agents > 0:
                    sample = sample.unsqueeze(0)  # Add extra dimension
                    repeated_sample = sample.repeat(max_agents // agents,  1)
                    padded_att_in[i, :agents] = repeated_sample[:agents]  # Assign values

                    print("shape padded_att_in: ", np.shape(padded_att_in[i, :agents]))
                    print("shape repeated_sample[:agents]: ", np.shape(repeated_sample[:agents]))
                    
            mask = torch.arange(max_agents, device=device) < torch.tensor(
                agents_per_sample, device=device
            )[:, None]

            padded_att_in = padded_att_in.permute(1, 0, 2)
            mask_inverted = ~mask
            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in, padded_att_in, padded_att_in, key_padding_mask=mask_inverted
            )
            padded_att_in_reswapped = padded_att_in_swapped.permute(1, 0, 2)

            for i in range(len(agents_per_sample)):
                agents = agents_per_sample[i]
                if agents > 0:
                    att_out_batch.append(padded_att_in_reswapped[i, :agents])

        else:
            att_in = torch.split(att_in, agents_per_sample)
            for sample in att_in:
                if sample.size(1) > 0:
                    att_out, _ = self.multihead_attention(
                        sample, sample, sample
                    )
                    att_out_batch.append(att_out.squeeze(dim=1))

        return att_out_batch
    
class STP_GR_Net_Attention(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_Attention, self).__init__()
        self.args = args
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])

        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # Attention module:
        self.attention = MultiheadSelfAttention(self.args['encoder_size'], self.args['num_heads'])

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

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
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
        Hist_Enc = self.attention(Hist_Enc, data_pyg.batch) # Apply attention to LSTM output
        tar_GAT_Enc = self.GAT_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        enc = torch.cat([Hist_Enc[target_index], tar_GAT_Enc], dim=-1)
        fut_pred = self.decode(enc)
        return fut_pred
''' 
class MultiheadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention, self).__init__()
        self.args = args
        self.latent_size = 128
        self.num_heads=4

        self.multihead_attention = nn.MultiheadAttention(self.latent_size, self.num_heads)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)
            
            print('max agents to handle: ', max_agents)


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
'''
############################################
#  END MODELS LOADER
############################################

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

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
        print('cgc_feature-Layer1: ', cgc_feature.size())

        cgc_feature = self.cgc_conv2(cgc_feature, edge_index, edge_attr)
        print('cgc_feature-layer2: ', cgc_feature.size())
        target_cgc_feature = cgc_feature[target_index]

        CGCNN_Enc = self.leaky_relu(self.nbrs_fc(target_cgc_feature))

        return CGCNN_Enc

    #def decode(self, enc):
    #    batch_size, out_length, decoder_size = enc.size()
    #    enc = enc.view(-1, decoder_size)
    #    h_dec, _ = self.dec_rnn(enc)
    #    fut_pred = self.op(h_dec)
    #    fut_pred = fut_pred.view(batch_size, out_length, 2)
    #    return fut_pred

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

        fut_pred = self.decode(torch.cat([Hist_Enc.unsqueeze(1), tar_CGCNN_Enc.unsqueeze(1).repeat(1, self.args['out_length'], 1)], dim=2))
        print('fut_pred: ', fut_pred.shape)
        return fut_pred

class STP_GR_Net_NOW(torch.nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_NOW, self).__init__()
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

    
    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        h_dec = h_dec.reshape(-1, self.args['decoder_size'])  # Reshape h_dec
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


#------------ ORIGINAL APPROACH    ---------
from stp_g_model import STP_G_Net

class STP_GR_Network(STP_G_Net):
    def __init__(self, args):
        super(STP_GR_Network, self).__init__(args)
        self.args = args
        # decoder based on RNN
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)

    def forward(self, data_pyg):

        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n single TP or multiple TP? \n\n')

        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        # Interaction
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)

        # get the lstm features of target vehicles
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]

        # Combine Individual and Interaction features
        enc = torch.cat((fwd_tar_LSTM_Enc, fwd_tar_GAT_Enc), 1)
        # Decode
        fut_pred = self.decode(enc)
        # get a Bi-LSTM decoder in two directions:



        return fut_pred

#----------------------------
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected




class STP_GR_Net_GATv2(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_GATv2, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size']//2, heads=4)
        self.gat_conv2 = GATConv(self.args['encoder_size']//2, self.args['encoder_size'], heads=4)

        self.nbrs_fc = nn.Linear(self.args['encoder_size'], 1*self.args['encoder_size'])
        self.dec_rnn = nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        
        self.op = nn.Linear(self.args['decoder_size'], 2)
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
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
        print('node_matrix: ', node_matrix)
        
        cgc_feature = self.gat_conv1(node_matrix, edge_index)
        print('cgc_features-level1: ', cgc_feature)
        print('size-cgc_features-level1:', cgc_feature.size())
        cgc_feature = self.gat_conv2(cgc_feature, edge_index)
        print('cgc_features-level2: ', cgc_feature)

        target_cgc_feature = cgc_feature[target_index]

        CGCNN_Enc = self.leaky_relu(self.nbrs_fc(target_cgc_feature))

        return CGCNN_Enc

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    


    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)

        if data_pyg.edge_attr is not None:
            edge_attr = self.define_edge_attr(data_pyg)
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), edge_attr.float(), target_index)
        else:
            tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), None, target_index)

        fut_pred = self.decode(torch.cat([Hist_Enc.unsqueeze(1), tar_CGCNN_Enc.unsqueeze(1).repeat(1, self.args['out_length'], 1)], dim=2))
        print('fut_pred: ', fut_pred.shape)
        return fut_pred
    '''
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
        
        return fut_pred    
    
    '''
##################################
#  Transformer: 
# ################################    
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class multiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        ## Stack all weights matrices 1 ... h together for efficiency 
        ## in some implementations "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()  # Call the parameter initialization

    def _reset_parameters(self):
        ## Original Transformer initialization...
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
   
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from Linear Output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)       
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]       
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else: 
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.5):
        super().__init__()

        # Attention layer
        self.self_attn = multiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
    

class STP_GR_Net_GATv3(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_GATv3, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        # GATConv -- GATConv2
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATConv(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # previous...
        #self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=4)
        #self.gat_conv2 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=4)

        #self.nbrs_fc = nn.Linear(self.args['encoder_size'], 1*self.args['encoder_size'])
        self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads']) * (self.args['num_gat_heads'] - 1) * self.args['encoder_size'] + self.args['encoder_size'], 1 * self.args['encoder_size'])

        self.dec_rnn = nn.LSTM(self.args['encoder_size'], 
                                     self.args['decoder_size'], 2, 
                                     batch_first=True)
        
        self.tran_head=2
        self.dim_feedforward = 2 # hidden dim of linear net

        #self.transformer= TransformerEncoder(self.args['decoder_size'])
        self.transformer =  TransformerEncoder(num_layers=1,
                                              input_dim=self.args['decoder_size'],
                                              dim_feedforward=self.dim_feedforward,
                                              num_heads=8,
                                              dropout=0.1)

        #self.op = nn.Linear(self.args['decoder_size'], 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc
    
    def define_edge_attr(self, data_pyg):
        # Define edge_attr using distance between nodes
        edge_index = data_pyg.edge_index
        node_pos = data_pyg.pos
        #num_nodes = data_pyg.num_nodes
        edge_attr = torch.cdist(node_pos[:, edge_index[0]], node_pos[:, edge_index[1]], p=2.0)
        edge_attr = edge_attr.view(-1, 1).repeat(1, 2).view(-1)
        return edge_attr

    def CGCNN_Interaction(self, hist_enc, edge_index, target_index):
        node_matrix = hist_enc
        #print('node_matrix: ', node_matrix)
        cgc_feature = self.gat_conv1(node_matrix, edge_index)
        #print('cgc_features-level1: ', cgc_feature)
        #print('size-cgc_features-level1:', cgc_feature.size())
        cgc_feature = self.gat_conv2(cgc_feature, edge_index)
        #print('cgc_features-level2: ', cgc_feature)
        target_cgc_feature = cgc_feature[target_index]
        #print('target_cgc_feature: ', target_cgc_feature)
        # Add a linear layer to reduce dimensionality before passing to self.nbrs_fc
        target_cgc_feature = self.leaky_relu(self.nbrs_fc(target_cgc_feature))
        #target_cgc_feature = self.leaky_relu(self.nbrs_fc(target_cgc_feature))
        return target_cgc_feature

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        
        fut_pred = self.transformer(h_dec)
        #fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        # print the input info 
        print('Hist_Enc.size: ', Hist_Enc.size())
        print('data_pyg.edge_index.size: ',  data_pyg.edge_index.size())
        print('target_index.size: ', target_index.size())
        tar_CGCNN_Enc = self.CGCNN_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        print('tar_CGCNN_Enc: ', tar_CGCNN_Enc)
        print('tar_CGCNN_Enc.size: ', tar_CGCNN_Enc.size())
        fut_pred = self.decode(tar_CGCNN_Enc)
        print('fut_pred: ', fut_pred.shape)
        return fut_pred  


from torch_geometric.nn import GCNConv
#from torch_geometric.nn import TransformerEncoder

class STP_GR_Net_GCNv4(nn.Module):
    '''
    This Graph network uses the Graph Convoltuion Network to predict trajectory
    -- conducted tests: 
        1. The first test to be conducted is to use the fully connected layer to carry out prediction
        2. The second test to be conducted is to use a Multi-Head Attention Layer instead of a fully connected layer for 
           the prediction.
        3. The third experiment to be conducted is to use a graph neural network layer to perform prediction. 
    '''
    def __init__(self, args):
        super(STP_GR_Net_GCNv4, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        # GCNConv -- GCNConv2
        self.gcn_conv1 = GCNConv(self.args['encoder_size'], self.args['encoder_size'])
        #print('gcn_conv_1_size: ', self.gcn_conv1)
        self.gcn_conv2 = GCNConv(self.args['encoder_size'], self.args['encoder_size'])
        print('gcn_conv_2_size: ', self.gcn_conv2)

        self.nbrs_fc = torch.nn.Linear(self.args['encoder_size'], 1 * self.args['encoder_size'])
        self.dec_rnn = nn.LSTM(self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        
        ## this is the parameters of the Transformer 
        #self.dim_feedforward = 2  # hidden dim of linear net
        #self.transformer = TransformerEncoder(num_layers=1, input_dim=self.args['decoder_size'], dim_feedforward=self.dim_feedforward, num_heads=2, dropout=0.1)
        ## This is the fully connected layer
        self.op = nn.Linear(self.args['decoder_size'], 2)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc

    def GCN_Interaction(self, hist_enc, edge_index, target_index):
        node_matrix = hist_enc
        gcn_feature = self.gcn_conv1(node_matrix, edge_index)
        gcn_feature = self.gcn_conv2(gcn_feature, edge_index)
        target_gcn_feature = gcn_feature[target_index]
        target_gcn_feature = self.leaky_relu(self.nbrs_fc(target_gcn_feature))
        return target_gcn_feature

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        
        # the 1st layer is a transformer
        #fut_pred = self.transformer(h_dec)
        # the 2nd layer is a fully connected layer
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        tar_GCN_Enc = self.GCN_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        fut_pred = self.decode(tar_GCN_Enc)
        return fut_pred

################################################
# EGCN 
################################################
from torch_geometric.nn import EGConv

class STP_GR_Net_EGConv(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_EGConv, self).__init__()
        self.args = args
        self.ip_emb = nn.Linear(2, self.args['input_embedding_size'])
        self.enc_rnn = nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.dyn_emb = nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        
        # EGConv layers
        self.egconv1 = EGConv(self.args['encoder_size'], self.args['encoder_size'])
        self.egconv2 = EGConv(self.args['encoder_size'], self.args['encoder_size'])

        self.nbrs_fc = torch.nn.Linear(self.args['encoder_size'], 1 * self.args['encoder_size'])
        self.dec_rnn = nn.LSTM(self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        
        self.op = nn.Linear(self.args['decoder_size'], 2)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def LSTM_Encoder(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc

    def GCN_Interaction(self, hist_enc, edge_index, target_index):
        node_matrix = hist_enc
        egconv_feature = self.egconv1(node_matrix, edge_index)
        egconv_feature = self.egconv2(egconv_feature, edge_index)
        target_egconv_feature = egconv_feature[target_index]
        target_egconv_feature = self.leaky_relu(self.nbrs_fc(target_egconv_feature))
        return target_egconv_feature

    def decode(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred
    
    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        tar_EGConv_Enc = self.GCN_Interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        fut_pred = self.decode(tar_EGConv_Enc)
        return fut_pred

################################################
# END implementation
################################################
class Encoder(nn.Module):
    def __init__(self, input_size, encoder_size, dyn_embedding_size):
        super(Encoder, self).__init__()
        self.ip_emb = nn.Linear(2, input_size)
        self.enc_rnn = nn.GRU(input_size, encoder_size, 1, batch_first=True)
        self.dyn_emb = nn.Linear(encoder_size, dyn_embedding_size)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, Hist):
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1], Hist_Enc.shape[2]))))
        return Hist_Enc


class GATInteraction_Enhanced(nn.Module):
    def __init__(self, encoder_size, num_gat_heads, concat_heads):
        super(GATInteraction_Enhanced, self).__init__()
        self.gat_conv1 = GATConv(encoder_size, encoder_size, heads=num_gat_heads, concat=concat_heads, dropout=0.0)
        self.gat_conv2 = GATConv(int(concat_heads) * (num_gat_heads - 1) * encoder_size + encoder_size, encoder_size, heads=num_gat_heads, concat=concat_heads, dropout=0.0)
        self.nbrs_fc = torch.nn.Linear(int(concat_heads) * (num_gat_heads - 1) * encoder_size + encoder_size, 1 * encoder_size)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, hist_enc, edge_index, target_index):
        node_matrix = hist_enc
        edge_attr = to_dense_adj(edge_index, edge_attr=None)[0]
        cgc_feature = self.gat_conv1(node_matrix, edge_attr)
        cgc_feature = self.gat_conv2(cgc_feature, edge_attr)
        target_cgc_feature = cgc_feature[target_index]
        target_cgc_feature = self.leaky_relu(self.nbrs_fc(target_cgc_feature))
        return target_cgc_feature


class Decoder(nn.Module):
    def __init__(self, encoder_size, decoder_size, out_length):
        super(Decoder, self).__init__()
        self.dec_rnn = nn.LSTM(encoder_size, decoder_size, 2, batch_first=True)
        self.op = nn.Linear(decoder_size, 2)

    def forward(self, enc):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        h_dec, _ = self.dec_rnn(enc)
        fut_pred = self.op(h_dec)
        return fut_pred


class STP_GR_Net_GATv3_Enhanced(nn.Module):
    def __init__(self, args):
        super(STP_GR_Net_GATv3_Enhanced, self).__init__()
        self.args = args
        self.encoder = Encoder(args['input_embedding_size'], args['encoder_size'], args['dyn_embedding_size'])
        self.gat_interaction = GATInteraction_Enhanced(args['encoder_size'], args['num_gat_heads'], args['concat_heads'])
        self.decoder = Decoder(args['encoder_size'], args['decoder_size'], args['out_length'])

    def forward(self, data_pyg):
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch == i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')

        Hist_Enc = self.encoder(data_pyg.x)
        tar_CGCNN_Enc = self.gat_interaction(Hist_Enc, data_pyg.edge_index.long(), target_index)
        fut_pred = self.decoder(tar_CGCNN_Enc)
        return fut_pred









################################################
#
################################################


class STP_Dynamics_Net(STP_Base_Net):
    def __init__(self, args):
        super(STP_Dynamics_Net, self).__init__(args)
        self.dec_rnn = torch.nn.LSTM(self.args['dyn_embedding_size'], self.args['decoder_size'], 2, batch_first=True)

    def forward(self, data_pyg):
        
        # get target vehicles' index first
        ########################################################################
        # for single TP
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        ########################################################################
       
        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        
        # get the lstm features of target vehicles
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]

        # Decode
        fut_pred = self.decode(fwd_tar_LSTM_Enc)
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
        #print('Observation dimension:', np.shape(data.x) )
        #print('Future dimension:', np.shape(data.y) )
    
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


#############################
# RMSE Lateral and Longitudinal
#############################
def maskedMSETest_lat(y_pred, y_gt, mask, separately=False):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    #muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    #y = y_gt[:, :, 1]
    out =torch.pow(x - muX, 2) #torch.pow(y - muY, 2) # #+ torch.pow(y - muY, 2) 
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts

def maskedMSETest_long(y_pred, y_gt, mask, separately=False):
    acc = torch.zeros_like(mask)
    #muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    #x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(y - muY, 2) #torch.pow(x - muX, 2) #+ torch.pow(y - muY, 2) 
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts


#############################
#
#############################
def val_a_model(model_to_val):
    model_to_val.eval()
    #lossVals = torch.zeros(10)
    #counts = torch.zeros(10)
    
    # Initialise var
    lossVals = torch.zeros(25)
    counts = torch.zeros(25)

    lossVal_lat = torch.zeros(25)
    counts_lat = torch.zeros(25)

    lossVal_lon = torch.zeros(25)
    counts_lon = torch.zeros(25)
    
    with torch.no_grad():
        print('Validation...')
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
            #print('mask_dim: ', np.shape(op_mask))
            l, c = maskedMSETest(fut_pred, ff, op_mask)
            ll_lat, cc_lat = maskedMSETest_lat(fut_pred, ff, op_mask,separately=False)

            ll_lon, cc_lon = maskedMSETest_long(fut_pred, ff, op_mask,separately=False )
            #--Detach
            lossVals +=l.detach()
            counts += c.detach()
            #############
            lossVal_lat+=ll_lat.detach()
            counts_lat+=cc_lat.detach()
            #############
            lossVal_lon+=ll_lon.detach()
            counts_lon+=cc_lon.detach()
            #############
            # 
            # 
    rmseOverall_lat=torch.pow(lossVal_lat / counts_lat,0.5) *0.3048    
    
    rmseOverall_lon=torch.pow(lossVal_lon / counts_lon,0.5) *0.3048   

    rmseOverall=torch.pow(lossVals / counts,0.5) *0.3048
    # print the prediction outcome :: TODO:: keep
    print("Prediction RMSE: ", torch.pow(lossVals / counts,0.5) *0.3048)   
    print("Prediction RMSE over 5s: ", rmseOverall[4::5])
    # mean error 
    print("RMSE (m):", rmseOverall[4::5],  "Mean= : ", np.array(rmseOverall[4::5]).mean())
    
    # lateral & long
    # 1.lateral
    print("Prediction RMSE lateral: ", torch.pow(lossVal_lat / counts_lat,0.5) *0.3048)   
    print("Prediction RMSE over 5s for lateral prediction: ", rmseOverall_lat[4::5])
    # mean error 
    print("RMSE (m) lateral:", rmseOverall_lat[4::5],  "Mean= : ", np.array(rmseOverall_lat[4::5]).mean())
    # 1.longitudinal
    print("Prediction RMSE longitudinal: ", torch.pow(lossVal_lon / counts_lon,0.5) *0.3048)   
    print("Prediction RMSE over 5s for longitudinal prediction: ", rmseOverall_lon[4::5])
    # mean error 
    print("RMSE (m) longitudinal:", rmseOverall_lon[4::5],  "Mean= : ", np.array(rmseOverall_lon[4::5]).mean())

    #print("Prediction NLL: ", )
    return torch.pow(lossVals / counts,0.5) *0.3048

def test_a_model(model_to_test):
    model_to_test.eval()
    #lossVals = torch.zeros(10)
    #counts = torch.zeros(10)

    lossVals = torch.zeros(25)
    counts = torch.zeros(25)
    
    with torch.no_grad():
        print('Testing...')
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
            #print('mask_dim: ', np.shape(op_mask))

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
    args['random_seed'] = 1 # increase the number of seeds
    args['input_embedding_size'] = 16 # if args['single_or_multiple'] == 'single_tp' else 32
    args['encoder_size'] = 32#32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['decoder_size'] = 32#64 #default = 64 # if args['single_or_multiple'] == 'single_tp' else 128 # 128 256
    args['dyn_embedding_size'] = 32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128

    args['edge_filters']=32

    args['train_epoches'] = 100

    args['num_gat_heads'] = 3 #3 #default=3
    args['concat_heads'] = True # False # True
    
    args['in_length'] = cmd_args.histlength
    args['out_length'] = cmd_args.futlength
    
    args['single_or_multiple'] = 'single_tp' # or multiple_tp single_tp
    args['date'] = date.today().strftime("%b-%d-%Y")
    args['batch_size'] = 16 if args['single_or_multiple'] == 'single_tp' else 128 # default is 16

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
    
    # Network selection 
    '''
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

    '''
    
    #train_net = STP_GR_Net_4(args)
    #train_net = STP_GR_Network(args)
    #train_net = STP_G_Net(args) # default model
    # -- The model works with a GRU Encoder, a GAT Interaction and a Transformer Decoder:
    #train_net = STP_GR_Net_GATv3(args)
    # -- The model works with a GRU Encoder, a CGN Interaction Encoder and a Fully Connected layer/ Transformer Decoder
    
    ##train_net = STP_GR_Net_GCNv4(args) # Graph Convolution network
    train_net = STP_GR_Net_EGConv(args)
    train_net.to(args['device'])

    '''
    # should be fixed!
    #train_net = STP_GR_Net_Attention(args)
    #1. First model
    train_net = STP_GR_Net_2(args)
    #2. Second model
    train_net3 = STP_G_Net(args)  # with dynamic only
    #3. Third model
    Train_net1 = STP_R_Net(args) # TODO:: decativate 
    

    
    #####################################
    # START : LOAD THE NETWORK
    #####################################
    # Initialize network


    #train_net = STP_Net2(args) # NGSIM :: GATv2 + Linear
    
    ##train_net = STP_GR_Net_2(args)


    train_net.to(args['device'])
    '''
    
    # count the number of parameters
    pytorch_total_params = sum(p.numel() for p in train_net.parameters())

    print('number of parameters: {}'.format(pytorch_total_params))
    print('NET: ', train_net)

    parent_parser.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))
   
    optimizer = torch.optim.Adam(train_net.parameters(),lr=0.004) 
    scheduler = MultiStepLR(optimizer, milestones=[1,2,3,6,20,30], gamma=0.5)

    if args['single_or_multiple'] == 'multiple':
        optimizer = torch.optim.Adam(train_net.parameters(),lr=0.004) # lr 0.0035, batch_size=4 or 8.
        #scheduler = MultiStepLR(optimizer, milestones=[1,2,3,6,20,30], gamma=0.5)
        scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6], gamma=1.0)
    

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
    trainDataloader = DataLoader(train_set, 
                                 batch_size=args['batch_size'], 
                                 num_workers=0, 
                                 pin_memory=True,
                                 drop_last=False,
                                 shuffle=True)
    valDataloader = DataLoader(val_set, 
                               batch_size=args['batch_size'], 
                               num_workers=0, 
                               pin_memory=True,
                               drop_last=False,
                               shuffle=False)
    
    # tic tac
    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []

    min_val_loss = 1000.0

    for ep in range(1, args['train_epoches']+1):
        print('epochs num: ', ep)
        
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
