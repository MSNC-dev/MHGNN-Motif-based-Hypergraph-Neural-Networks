# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys, os
import torch
import re
import string
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def build_k_hop_adjacency(adj: torch.Tensor, k: int):  # Build a k-hop adjacency matrix.
    if k==1:
        KIM=torch.where(torch.matrix_power(adj, k)>0,1,0)
    else:
        KIM=(torch.where(torch.matrix_power(adj, k)>0,1,0)-torch.where(torch.matrix_power(adj, k-1)>0,1,0))
    return KIM


def repeat_indices(num_repeats, max_num, device):  # Repeat each index num_repeats times, e.g., (2, 4) -> [0, 0, 1, 1, 2, 2, 3, 3].
    nums = torch.arange(0, max_num).to(device)
    repeated_nums = nums.repeat_interleave(num_repeats)
    return repeated_nums

def build_knn_hyperedge_index(emb: torch.Tensor, k: int = 3, p: int = 2):  # Build a KNN hyperedge index from node embeddings.
    batch = emb.shape[0]
    t = torch.cdist(emb, emb, p=p)  # Compute pairwise distances between node embeddings.
    nodes = t.argsort(dim=-1)[:, :, :k + 1].reshape(-1)  # Select each node and its k nearest neighbors.
    nodes += repeat_indices((k + 1) * emb.shape[1], batch, emb.device) * emb.shape[1]  # Offset node IDs for each batch graph.

    edges = torch.zeros(emb.shape[1]*(k+1)*emb.shape[0],device=emb.device)
    edges += repeat_indices(k + 1, batch * emb.shape[1], emb.device)  # Each hyperedge contains k + 1 nodes, with one hyperedge per node per batch.

    return torch.stack([nodes,edges],dim=0).long()


def compute_knn_node_indices(emb: torch.Tensor, p: int = 2):  # Return sorted KNN node indices for inspection.
    t = torch.cdist(emb,emb,p=2)
    nodes = t.argsort(dim=-1)  # Sort nodes by distance, including the node itself.
    return nodes



def build_k_hop_hyperedge_index(adj: torch.Tensor, k: int):  # Build a k-hop hyperedge index with shape (2, n).
    if k==1:
        KIM=torch.unique(torch.where(torch.matrix_power(adj, k)>0,1,0),dim=1)
    else:
        KIM=torch.unique((torch.where(torch.matrix_power(adj, k)>0,1,0)-torch.where(torch.matrix_power(adj, k-1)>0,1,0)),dim=1)  # Exclude nodes already covered by the (k-1)-hop neighborhood.
    e2n=KIM.sum(dim=0).sum(dim=0)  # Count all node-hyperedge incidence pairs.
    pairwise=torch.zeros((2,e2n)).long()  # Shape: (2, n).
    index=0
    for i in range(KIM.shape[0]):
        for j in range(KIM.shape[1]):
            if KIM[i][j]==1:
                pairwise[0][index]=i
                pairwise[1][index]=j
                index+=1

    return pairwise



def compute_hypergraph_propagation_matrix(H, variable_weight=False):
    if type(H) != list:
        return _compute_hypergraph_propagation_matrix(H, variable_weight)
    G = []
    for sub_H in H:
        G.append(compute_hypergraph_propagation_matrix(H, variable_weight))
    return G
def _compute_hypergraph_propagation_matrix(H, variable_weight=False):
    n_edge = H.shape[1]
    W = torch.ones(n_edge, device=H.device)
    DV = torch.sum(H * W, dim=1)
    DE = torch.sum(H, dim=0)
    W = torch.diag(W)
    invDE = torch.diag(torch.pow(DE, -1))
    invDV = torch.diag(torch.pow(DV, -1))
    HT = H.T

    if variable_weight:
        invDV_H = invDV @ H
        invDE_HT_DV2 = invDE @ HT 
        return invDV_H, W, invDE_HT_DV2
    
    G = invDV @ H @ W @ invDE @ HT 
    return G

# get laplace matrix
def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    '''
    d_mat = torch.bmm(adj, adj.permute(0, 2, 1))
    d_mat = torch.where(i_mat>0, d_mat, i_mat)
    print('d_mat version 1', d_mat)
    '''
    d_mat_in = torch.sum(adj, dim=1)
    d_mat_out = torch.sum(adj, dim=2)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    #d_mat = torch.pow(d_mat, -0.5) if is 1/2
    d_mat = torch.pow(d_mat, -1)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    #laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat
 


 # define peak area in ground truth data
def peak_error(y_true_states, y_pred_states, threshold): 
    # masked some low values (using training mean by states)
    y_true_states[y_true_states < threshold] = 0
    mask_idx = np.argwhere(y_true_states <= threshold)
    for idx in mask_idx:
        y_pred_states[idx[0]][idx[1]] = 0
    # print(y_pred_states,np.count_nonzero(y_pred_states),np.count_nonzero(y_true_states))
    
    peak_mae_raw = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    # peak_mae_std = np.std(peak_mae_raw)
    return peak_mae


    
def normalize_adj2(adj):
    """Symmetrically normalize adjacency matrix."""
    # print(adj.shape)
    # adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

 
def normalize(mx):
    """Row-normalize sparse matrix  (normalize feature)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
