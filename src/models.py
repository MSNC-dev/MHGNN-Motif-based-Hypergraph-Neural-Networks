# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch_geometric.nn.conv import HypergraphConv
from time import time

from utils import *
from layers import *
from ablation import baseline
from ablation import without_motifs
from ablation import all_gcn

class MHGNN(nn.Module):
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
        self.KNNhop = args.KNNhop
        self.knn_nodes = args.knn_nodes
        self.motifs = data.motifs
        self.motifs_num = []
        if True:
            self.khop_num = args.khop_num
            self.G=[] #store hypergraphs shape as (n,n) 
            for j in range(self.khop_num):  # Build motif-related hyperedge indices with shape (2, n): row 0 stores node IDs and row 1 stores hyperedge IDs.
                count = 0
                for i in range(8):  # Eight motif types.
                #for i in range(2,6):
                    #if i == 1 or i == 7:
                        #continue
                    a = build_k_hop_hyperedge_index(self.motifs[i],j+1).to(self.adj.device)
                    if not torch.any(a):  # Skip empty hypergraphs.
                        continue
                    self.G.append(a)
                    count+=1
                self.motifs_num.append(count)  # Record the number of non-empty motif graphs for each motif-hop.
            
            for j in range(self.khop_num):  # Build global hyperedge indices.
                self.G.append(build_k_hop_hyperedge_index(self.adj,j+1).to(self.adj.device))
        
        self.IM = []  # store Incidence matrix shape as (2, n)
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


        self.HGNNBlocks = nn.ModuleList([HypergraphConv(in_channels=self.hidR, out_channels=self.hidR, dropout=self.droprate) for _ in range((self.khop_num*2 + self.KNNhop) * self.n)])  # Allocate khop_num motif hypergraph layers, khop_num global hypergraph layers, and one optional KNN hypergraph layer.
        self.act = nn.ELU()

        #self.GCNBlock1 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
        #self.GCNBlock2 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)

        
        self.attr_motifs = nn.ModuleList([motif_Attention(self.adj.shape[0], self.hidR, i) for i in self.motifs_num])  # Fuse multiple 1/2/3-hop motif hypergraph features into one motif feature.
        self.attr_local = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num)  # Fuse 1/2/3-hop motif hypergraph features into one local hypergraph feature.
        self.attr_global = motif_Attention(self.adj.shape[0], self.hidR, self.khop_num + self.KNNhop)  # Fuse 1/2/3-hop global hypergraph features and the optional KNN feature.
        self.attr_feat = motif_Attention(self.adj.shape[0], self.hidR, 2)  # Fuse local and global hypergraph features.
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
        adj = torch.zeros((x.shape[-1],x.shape[-1]))
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
        
        # Graph Convolution Network
        node_state_ = feat_emb

        IM = self.IM[:]
        if self.KNNhop:  # Build the KNN hypergraph.
            if self.knn_nodes == -1:
                IM.append(build_knn_hyperedge_index(feat_emb,int(np.log2(adj.shape[0]))))
            else:
                IM.append(build_knn_hyperedge_index(feat_emb,self.knn_nodes))
            # self.output_knn = compute_knn_node_indices(feat_emb)

        node_state_list=[]
        # Motif hypergraph feature propagation.
        for hop in range(self.khop_num):
            for i in range(self.motifs_num[hop]):
                node_state = node_state_
                idx = sum(self.motifs_num[:hop]) + i
                for n in range(self.n):  # Reshape node features to merge all batch graphs into one graph.
                    node_state = self.HGNNBlocks[hop*self.n+n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[idx][:,:batch_size*self.G[idx].shape[1]])
                    node_state = self.dropout(node_state)
                    node_state = self.act(node_state) 
                node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))
        # Global hypergraph feature propagation.
        for hop in range(self.khop_num):
            node_state = node_state_
            for n in range(self.n):
                node_state = self.HGNNBlocks[(hop+self.khop_num)*self.n+n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[sum(self.motifs_num)+hop][:,:batch_size*self.G[sum(self.motifs_num)+hop].shape[1]])
                node_state = self.dropout(node_state)
                node_state = self.act(node_state) 
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))
        # KNN hypergraph feature propagation.
        if self.KNNhop:
            node_state = node_state_
            for n in range(self.n):
                node_state = self.HGNNBlocks[-1-n](x=node_state.reshape(-1,self.hidR), hyperedge_index=IM[-1])
                node_state = self.dropout(node_state)
                node_state = self.act(node_state)
            node_state_list.append(node_state.reshape(batch_size,-1,self.hidR))
    

        # Final prediction
        local_list = []
        for i in range(self.khop_num):
            local_list.append(self.attr_motifs[i](node_state_list[sum(self.motifs_num[:i]):sum(self.motifs_num[:i+1])]))  # Fuse motif features within each 1/2/3-hop group.
        local_feat = self.attr_local(local_list)  # Fuse 1/2/3-hop motif hypergraph features.
        global_feat = self.attr_global(node_state_list[sum(self.motifs_num):])  # Fuse 1/2/3-hop global hypergraph features and the KNN feature.

        node_state = self.attr_feat([local_feat,global_feat])
        node_state = torch.cat([node_state,feat_emb],dim=-1)
        res = self.output(node_state).squeeze(2)
        
        # self.output_similarity = [torch.cdist(node_state[i],node_state[i],p=2) for i in range(batch_size)] # Euclid distance

        # X_normalized = F.normalize(node_state, dim=2)
        # self.cosine_similarity = torch.einsum('bif,bjf->bij', X_normalized, X_normalized) # cosine_similarity

        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None
        return res, imd
