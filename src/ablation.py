# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils import *
from torch.autograd import Variable
import sys
import math
from time import time
from torch_geometric.nn import HypergraphConv
from layers import motif_Attention
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

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))

        return adj

def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    d_mat = torch.pow(d_mat, -0.5)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat

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


class without_motifs(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.m = data.m
        self.w = args.window
        self.n_layer = args.n_layer
        self.droprate = args.dropout
        self.hidR = args.hidR
        self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.res = args.res
        self.dropout = nn.Dropout(self.droprate)
        self.batch = args.batch

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        
        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = data.degree_adj
        self.s_enc = nn.Linear(1, self.hidR)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.KNNhop = args.KNNhop
        self.motifs = data.motifs
        self.motifs_num = []
        if True:
            
            self.khop_num = args.khop_num
            self.G=[]
            for j in range(self.khop_num):
                count = 0
                for i in range(8):
                    a = build_k_hop_hyperedge_index(self.motifs[i],j+1).to(self.adj.device)
                    if not torch.any(a):
                        continue
                    self.G.append(a)
                    count+=1
                self.motifs_num.append(count)
            
            for j in range(self.khop_num):
                self.G.append(build_k_hop_hyperedge_index(self.adj,j+1).to(self.adj.device))
        
        self.IM = []                                                                                                            
        for hop in range(len(self.G)):
            if self.G[hop].shape == (2,0):
                self.IM.append(torch.tensor([[],[]]).long().to(self.adj.device))
                continue
            node_nums = self.G[hop][0].max() + 1
            edge_nums = self.G[hop][1].max() + 1
            length = self.G[hop].shape[1]
            t = self.G[hop].repeat(1,self.batch)
            for i in range(self.batch):
                t[0][i*length:(i+1)*length] += node_nums*i                                                                              
                t[1][i*length:(i+1)*length] += edge_nums*i
            self.IM.append(t)

        self.HGNNBlocks = nn.ModuleList([HypergraphConv(in_channels=self.hidR, out_channels=self.hidR, dropout=self.droprate) for _ in range((self.khop_num*2 + self.KNNhop) * self.n)])


        self.attr_motifs = nn.ModuleList([motif_Attention(self.adj.shape[0], self.hidR, i) for i in self.motifs_num])
        self.attr_local = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num)
        self.attr_global = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num + self.KNNhop)
        self.attr_feat = motif_Attention(self.adj.shape[0], self.hidR, 2)
        self.output_knn = None 
        self.output_similarity = None
        self.cosine_similarity = None
        # prediction
        if self.res == 0:
            self.output = nn.Linear(self.hidR*2, 1)
        else:
            self.output = nn.Linear(self.hidR*(self.n+1), 1)

        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, index, isEval=False):
        #print(index.shape) batch_size
        batch_size = x.shape[0] # batchsize, w, m
         
        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(x)
        
        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        #d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc
        
        adj = torch.zeros((x.shape[-1],x.shape[-1]))


        # Graph Convolution Network
        node_state_ = feat_emb
        IM = self.IM[:]
        if self.KNNhop:
            IM.append(build_knn_hyperedge_index(feat_emb,int(np.log2(adj.shape[0]))))
            # self.output_knn = compute_knn_node_indices(feat_emb)

        
        node_state_list=[]
        #local_emb hgnn
        #global hgnn
        for hop in range(self.khop_num):
            node_state = node_state_
            for n in range(self.n):
                node_state = self.HGNNBlocks[(hop+self.khop_num)*self.n+n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[sum(self.motifs_num)+hop][:,:batch_size*self.G[sum(self.motifs_num)+hop].shape[1]])
                node_state = self.dropout(node_state)
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))
        #knn
        if self.KNNhop:
            node_state = node_state_
            for n in range(self.n):
                node_state = self.HGNNBlocks[-1-n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[-1])
                node_state = self.dropout(node_state)
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))


        # Final prediction
        global_feat = self.attr_global(node_state_list)
        node_state = torch.cat([global_feat,feat_emb],dim=-1)
        res = self.output(node_state).squeeze(2)
        #self.output_similarity = [torch.cdist(node_state[i],node_state[i],p=2) for i in range(batch_size)]
        #X_normalized = F.normalize(node_state, dim=2)
        #self.cosine_similarity = torch.einsum('bif,bjf->bij', X_normalized, X_normalized)

        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None
        return res, imd

class all_gcn(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.m = data.m
        self.w = args.window
        self.n_layer = args.n_layer
        self.droprate = args.dropout
        self.hidR = args.hidR
        self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.res = args.res
        self.dropout = nn.Dropout(self.droprate)
        self.batch = args.batch

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        
        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = data.degree_adj
        self.s_enc = nn.Linear(1, self.hidR)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.KNNhop = args.KNNhop
        self.motifs = data.motifs
        self.motifs_num = []
        if True:
            self.khop_num = args.khop_num
            self.G=[]
            for j in range(self.khop_num):
                count = 0
                for i in range(8):
                #for i in range(2,6):
                    #if i == 1 or i == 7:
                        #continue
                    a = build_k_hop_adjacency(self.motifs[i],j+1).to(self.adj.device)
                    if not torch.any(a):
                        continue
                    self.G.append(a)
                    count+=1
                self.motifs_num.append(count)
            
            for j in range(self.khop_num):
                self.G.append(build_k_hop_adjacency(self.adj,j+1).to(self.adj.device))
        
        self.IM = []                                                                                                            


        self.HGNNBlocks = nn.ModuleList([HypergraphConv(in_channels=self.hidR, out_channels=self.hidR, dropout=self.droprate) for _ in range(self.KNNhop * self.n)])
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.khop_num * 2 * self.n)])
        #self.GCNBlock1 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
        #self.GCNBlock2 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)


        self.attr_motifs = nn.ModuleList([motif_Attention(self.adj.shape[0], self.hidR, self.motifs_num[i]) for i in range(self.khop_num)])
        self.attr_local = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num)
        self.attr_global = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num + self.KNNhop )
        self.attr_feat = motif_Attention(self.adj.shape[0], self.hidR, 3)
        self.output_knn = None 
        self.output_similarity = None
        self.cosine_similarity = None
        # prediction
        if self.res == 0:
            self.output = nn.Linear(self.hidR, 1)
        else:
            self.output = nn.Linear(self.hidR*(self.n+1), 1)

        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, index, isEval=False):
        #print(index.shape) batch_size
        batch_size = x.shape[0] # batchsize, w, m
         
        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(x)
        
        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        #d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc
        
        adj = torch.zeros((x.shape[-1],x.shape[-1]))


        # Graph Convolution Network
        
        node_state_ = feat_emb
        IM = self.IM[:]
        if self.KNNhop:
            IM.append(build_knn_hyperedge_index(feat_emb,int(np.log2(adj.shape[0]))))
            # self.output_knn = compute_knn_node_indices(feat_emb)
        
        laplace_adj = [getLaplaceMat(batch_size, self.m, self.G[i]) for i in range(len(self.G))]

        node_state_list=[]
        #local_emb gcn
        for hop in range(self.khop_num):
            for i in range(self.motifs_num[hop]):
                node_state = node_state_
                for n in range(self.n):
                    node_state = self.GNNBlocks[hop*self.n+n](node_state, laplace_adj[i+sum(self.motifs_num[:hop])])
                    node_state = self.dropout(node_state)
                node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))

        #global gcn
        for hop in range(self.khop_num):
            node_state = node_state_
            for n in range(self.n):
                node_state = self.GNNBlocks[(hop+self.khop_num)*self.n+n](node_state, laplace_adj[sum(self.motifs_num)+hop])
                node_state = self.dropout(node_state)
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))

        #knn

        if self.KNNhop:
            node_state = node_state_
            for n in range(self.n):
                node_state = self.HGNNBlocks[n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[-1])
                node_state = self.dropout(node_state)
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))

        # Final prediction
        local_list = []
        for i in range(self.khop_num):
            local_list.append(self.attr_motifs[i](node_state_list[sum(self.motifs_num[:i]):sum(self.motifs_num[:i+1])]))
        local_feat = self.attr_local(local_list)
        global_feat = self.attr_global(node_state_list[sum(self.motifs_num):])

        node_state = self.attr_feat([local_feat,global_feat,feat_emb])
        res = self.output(node_state).squeeze(2)
        # self.output_similarity = [torch.cdist(node_state[i],node_state[i],p=2) for i in range(batch_size)]
        # X_normalized = F.normalize(node_state, dim=2)
        # self.cosine_similarity = torch.einsum('bif,bjf->bij', X_normalized, X_normalized)

        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None
        return res, imd


class DotAtt(nn.Module):
    def __init__(self, attn_dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = self.dropout(attn)
        attn = self.softmax(attn)
        return attn



   

class baseline(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.m = data.m
        self.w = args.window
        self.n_layer = args.n_layer
        self.droprate = args.dropout
        self.hidR = args.hidR
        self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.res = args.res
        self.dropout = nn.Dropout(self.droprate)

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        #self.backbone = TemporalConvNet(num_inputs=self.w, num_channels=[self.hidR]*3, kernel_size=self.s, dropout=self.droprate)

        # prediction
        self.output = nn.Linear(self.hidR, 1)


        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, index, isEval=False):
        #print(index.shape) batch_size
        batch_size = x.shape[0] # batchsize, w, m
        temp_emb = self.backbone(x)
        res = self.output(temp_emb).squeeze(2)
        imd=None
        return res, imd
