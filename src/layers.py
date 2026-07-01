import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable



class motif_Attention(nn.Module):
    def __init__(self, num_nodes, hidden_dim, n=8):
        super(motif_Attention, self).__init__()
        self.e = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.b = nn.Parameter(torch.rand(num_nodes,hidden_dim))
        self.t = nn.Parameter(torch.rand(hidden_dim))
        self.n = n
        self.tanh = nn.Tanh()
        self.weight = None
    def forward(self, x_motif):
        batch, n, d = x_motif[0].shape
        x_motif = torch.stack(x_motif,dim=0)#[motif,batch,node,feature]
        att_weight = F.softmax(self.tanh(x_motif @ self.e + self.b) @ self.t, dim=0)
        out = torch.sum(x_motif * att_weight.unsqueeze(-1).expand(-1,-1,-1,d),dim=0)
        self.weight = att_weight
        
        return out
Attention = motif_Attention
'''class Attention(nn.Module):
    def __init__(self, num_nodes, hidden_dim, n=8):
        super(Attention, self).__init__()
        self.e = nn.Parameter(torch.rand(hidden_dim,hidden_dim)) 
        self.b = nn.Parameter(torch.rand(num_nodes,hidden_dim)) 
        self.t = nn.Parameter(torch.rand(hidden_dim)) 
        self.n = n
        self.tanh = nn.Tanh()
    def forward(self, x_motif):
        batch, n, d = x_motif[0].shape
        att_weight = [F.softmax(self.tanh(embed @ self.e + self.b) @ self.t, dim=-1) for embed in x_motif]#[motifs,batch,num_nodes]
        x_motif = torch.stack(x_motif, dim = 0)
        out = torch.zeros_like(x_motif[0])
        for i in range(n):
            node_embedding = x_motif[:,:,i,:]#[motifs,batch_size,features] for the node
            node_weight = torch.stack([aw[:,i] for aw in att_weight], dim=0)#[motifs,batch_size]
            node_weight = node_weight.unsqueeze(-1).repeat(1,1,node_embedding.shape[-1])
            out[:,i,:] = torch.sum(node_embedding * node_weight, dim=0)
        return out'''

    
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class ConvBranch(nn.Module):
    def __init__(self,
                 m,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_factor,
                 hidP=1,
                 isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        batch_size = x.shape[0]
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x
