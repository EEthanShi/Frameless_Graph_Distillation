#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''teacher and student models for frameless distillation paper'''




import argparse
import torch
from scipy import sparse
import numpy as np 
from torch_geometric.utils import get_laplacian, degree, remove_self_loops, to_networkx,add_remaining_self_loops
from utils import set_seed
import copy
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from numba import jit, prange
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) 

'''necessary functions'''

@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


def getFilters(FrameType):
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')
    return DFilters

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c



""" not sure which function to use, by default using get operator not operator 2"""
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d
"""modified version for Prof Junbin Gao"""
def get_operator2(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(Lev):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J - l) / a) * L) @ T0F - T0F
            d[j, l] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J - l)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l] += c[j][k] * TkF
        FD1 = d[0, l]

    return d

# function for pre-processing For no Chebyshev approximation: Slow
def get_operator1(L, DFilters, lambdas, eigenvecs, s, Lev):
    num_nodes = L.shape[0]
    lambdas[lambdas <= 0.0] = 0.0
    lambdas[lambdas > 2.0] = 2.0
    r = len(DFilters)   # DFiliters is a list of g functions
    J =  np.log(lambdas[-1] / np.pi) / np.log(s)  # dilation level to start the decomposition
    #a = np.pi / 2  # consider the domain of masks as [0, pi]
    d = dict()
    FD1 = 1.0
    for l in range(Lev):
        for j in range(r):
            d[j, l] = FD1 * DFilters[j](s**(- J - l) * lambdas)
        FD1 = d[0, l]

    d_list = list()
    for i in range(r):
        for l in range(Lev):
            print('Calculating Matrix...{0:2d}, {1:2d}'.format(i, l))
            d_list.append(np.matmul(eigenvecs, np.diag(d[i, l]) @ eigenvecs.T))

    print(FD1)
    return d_list

    


def sparse_eye(size,edge_index, value):
    """
    Returns the identity * value matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(value).expand(size)
    return torch.sparse.FloatTensor(indices, values, torch.Size([size, size])) 


def sparse_degree(size, edge_index, value):
    """
    Returns the degree * value matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=size)
    index_tensor = edge_index[0]
    deg = degree(index_tensor, size)
    values = deg.pow_(-1) * value
    return torch.sparse.FloatTensor(indices, values, torch.Size([size, size])) 






#                            '''teacher models'''

#---------------------------------------spatial UFG---------------------------

class SpatUFGConv(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, d_list, symadj,
                 edge_index,
                 bias=True, epsilon=0.):
        # epsilon controls the amount of perturbation to the low-pass and high
        #-pass adj matrix
        
        super(SpatUFGConv, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.d_list = d_list
        self.symadj = symadj.to(device)
        Ad_list = list()
        for ii in range(len(d_list)):
            if ii == 0:
                # low-pass
                
                Ad_list.append(symadj - sparse_degree(symadj.shape[0], edge_index, epsilon).to(symadj.device))
            else:
                # high-pass
                
                Ad_list.append(symadj + sparse_degree(symadj.shape[0], edge_index, epsilon).to(symadj.device))
        self.Ad_list = Ad_list
        self.device = symadj.device
        
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(in_features, out_features)) for i in range(len(d_list))]) 
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for ii in range(len(self.weights)):
            nn.init.xavier_uniform_(self.weights[ii])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        d_list = self.d_list
        Ad_list = self.Ad_list
        
        out = torch.zeros(x.shape[0], self.out_features, device=self.device)
        for ii in range(len(Ad_list)):
            temp = torch.sparse.mm(d_list[ii], x @ self.weights[ii])   
            temp = torch.sparse.mm(Ad_list[ii], temp)
            #temp = torch.mm(temp, self.weights[ii])
            temp = torch.sparse.mm(d_list[ii], F.relu(temp))  #w relu a w x w
            out += temp
            
        if self.bias is not None:
            out += self.bias
        
        return out
    
    
class SpatUFG(nn.Module):
    def __init__(self, num_features, nhid, num_classes, num_nodes, d_list, symadj,
                 edge_index,
                 n_layers=2, dropout=0.5, epsilon=0.):
        super(SpatUFG, self).__init__()
        self.n_layers = n_layers
               
        self.GConv1 = SpatUFGConv(num_features, nhid, num_nodes, d_list, symadj, edge_index, epsilon=epsilon)
        if n_layers > 2:
            self.layers = nn.ModuleList([ 
                SpatUFGConv(nhid, nhid, num_nodes, d_list, symadj, edge_index, epsilon=epsilon) 
                for ii in range(n_layers - 2)
                    ])
            
        self.GConv2 = SpatUFGConv(nhid, num_classes, num_nodes, d_list, symadj, edge_index, epsilon=epsilon)
        self.drop1 = nn.Dropout(dropout)
        
    def forward(self, x):

        x = self.GConv1(x.double())
        x = self.drop1(x)
        if self.n_layers > 2:
            for ii in range(self.n_layers - 2):
                x = self.layers[ii](x)
                x = self.drop1(x)
        x = self.GConv2(x)

        return F.log_softmax(x, dim=1)     
    
    
# --------------------------------simplified ufg-----------------------------  
        
    
    
    
    
    


class Simplified_UFGConv(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, d_list, symadj,
                 edge_index, num_layers,
                 bias=True, epsilon=0.):
       
        
        super(Simplified_UFGConv, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.d_list = d_list
        self.symadj = symadj.to(device)
        self.num_layers = num_layers 
        Ad_list = list()
        for ii in range(len(d_list)):
            if ii == 0:
                # low-pass
                
                Ad_list.append(symadj - sparse_degree(symadj.shape[0], edge_index, epsilon).to(symadj.device))
            else:
                # high-pass
                
                Ad_list.append(symadj + sparse_degree(symadj.shape[0], edge_index, epsilon).to(symadj.device))
        self.Ad_list = Ad_list
        self.device = symadj.device
        
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(in_features, out_features)) for i in range(len(d_list))]) 
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for ii in range(len(self.weights)):
            nn.init.xavier_uniform_(self.weights[ii])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        d_list = self.d_list
        Ad_list = self.Ad_list
        num_layers = self.num_layers
        
        out = torch.zeros(x.shape[0], self.out_features, device=self.device)
        for ii in range(len(Ad_list)):
            temp = torch.sparse.mm(torch.sparse.mm(d_list[ii].transpose(0,1),Ad_list[ii]),d_list[ii]).to(device)
            temp = torch.matrix_power(temp.to_dense(),num_layers)    
            temp = torch.sparse.mm(temp, x @ self.weights[ii])   #(waw)^layers xw
            out += temp
            
        if self.bias is not None:
            out += self.bias
        
        return out


class Simplifed_UFG(nn.Module):
    def __init__(self, num_features, num_classes, num_nodes, d_list, symadj,
                 edge_index,n_layers=1, dropout=0.5, epsilon=0.):
        super(Simplifed_UFG, self).__init__()
        self.n_layers = n_layers
      
        self.GConv2 = Simplified_UFGConv(
            num_features, 
            num_classes, 
            num_nodes, 
            d_list, 
            symadj, 
            edge_index, 
            n_layers,
            epsilon=epsilon)
        
        self.drop1 = nn.Dropout(dropout)
        
    def forward(self, x):

        x = self.GConv2(x.double())
        return F.log_softmax(x, dim=1)  




#--------------------------------''' student models'''------------------------
        
    


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
            norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "bn":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "ln":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "bn":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "ln":
                    self.norms.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = self.dropout(h).relu()
                if self.norm_type != "none":
                    h = self.norms[l](h)
        return h    
    


#-------------------------------------FMLPO-----------------------------------  
      
class FMLPO(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, MLP_layer=1, teacher_layer = 2,
                  dropout=.5, norm_type='none'):
        super().__init__()
        self.mlp_lowpass = nn.Linear(num_nodes, hidden_channels)
        self.mlp_highpass = nn.Linear(num_nodes, hidden_channels)
        self.mlpX = nn.Linear(in_channels, hidden_channels)
        self.mlp_second_econding = nn.Linear(hidden_channels, hidden_channels)
        self.atten = nn.Linear(hidden_channels * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier_lowpass = MLP(hidden_channels, hidden_channels, out_channels,MLP_layer, dropout, norm_type=norm_type)
        self.classifier_highpass = MLP(hidden_channels, hidden_channels, out_channels,MLP_layer, dropout, norm_type=norm_type)
        self.classifierX = MLP(hidden_channels, hidden_channels, out_channels, MLP_layer,dropout, norm_type=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.latent_predictor = MLP(in_channels, hidden_channels, hidden_channels,2,dropout, norm_type=norm_type)
        self.layers = teacher_layer
        

    def hidden_layer_imitation(self,low_pass_d,high_pass_d, X):
        """
        low_pass_d =  W_0j ^T A W_0J,  high_pass_d = W_rj ^T A  W_rJ        
        """    
        H_lp = self.mlp_lowpass(low_pass_d)  # 1st layer encode low pass waw 
        
        H_lp = self.dropout(H_lp).relu()
        
        
        H_hp = self.mlp_highpass(high_pass_d) # 1st layer encode high pass waw              
        
        H_hp = self.dropout(H_hp).relu()
        
        
        HX = self.mlpX(X.double()) # encode x   
        
        HX = self.dropout(HX).relu()
        
        H_0j = torch.cat((H_lp, HX), dim=1)
        H_rj = torch.cat((H_hp, HX), dim=1)
        high_d_alpha_0j = self.atten(H_0j).sigmoid()  # generate low pass alpha in d_0 dimension 
        high_d_alpha_rj = self.atten(H_rj).sigmoid()  # generate hihg pass alpha in d_0 dimension 
        high_y_0j = H_lp * high_d_alpha_0j.view(-1, 1) + HX * (1 - high_d_alpha_0j.view(-1, 1))
        high_y_rj = H_hp * high_d_alpha_rj.view(-1, 1) + HX * (1 - high_d_alpha_rj.view(-1, 1))
        y_reconstruct = high_y_0j+high_y_rj
       
        
        return  y_reconstruct
    '''this is used when l>2, as we need maintain the y_reconstruct by using nn.linear(hidden_dim, hidden_dim)==>second ecoding'''    
    def intermediate_node_embedding (self, low_pass_d,high_pass_d, y_reconstruct):  
  
        H_lp = self.mlp_lowpass(low_pass_d)  
        H_lp = self.dropout(H_lp).relu()
        H_hp = self.mlp_highpass(high_pass_d) 
        H_hp = self.dropout(H_hp).relu()  
        y_reconstruct = self.mlp_second_econding(y_reconstruct)
        y_reconstruct = self.dropout(y_reconstruct).relu()
        
        H_0j = torch.cat((H_lp, y_reconstruct), dim=1)
        H_rj = torch.cat((H_hp, y_reconstruct), dim=1)
        high_d_alpha_0j = self.atten(H_0j).sigmoid()  # generate low pass alpha in d_0 dimension 
        high_d_alpha_rj = self.atten(H_rj).sigmoid()  # generate hihg pass alpha in d_0 dimension 
        high_y_0j = H_lp * high_d_alpha_0j.view(-1, 1) + y_reconstruct * (1 - high_d_alpha_0j.view(-1, 1))
        high_y_rj = H_hp * high_d_alpha_rj.view(-1, 1) + y_reconstruct * (1 - high_d_alpha_rj.view(-1, 1))
        y_reconstruct = high_y_0j+high_y_rj

        
        return  y_reconstruct
        

    
    def output_layer_imtation(self,low_pass_d_square,high_pass_d_square, y_reconstruct):
       
        HX_second_layer = self.mlp_second_econding(y_reconstruct) # optional when layer <2: encode the first layer output. 
        
        HX_second_layer = self.dropout(HX_second_layer).relu()
        
        H_lp_square =  self.mlp_lowpass(low_pass_d_square)  
        
        H_lp_square = self.dropout(H_lp_square).relu()
        
        
        H_hp_square =  self.mlp_lowpass(low_pass_d_square)
        
        H_hp_square = self.dropout(H_hp_square).relu()
        
        
        H_0j_cancat = torch.cat((H_lp_square, HX_second_layer), dim=1)
        H_rj_concat = torch.cat((H_hp_square, HX_second_layer), dim=1)  
        
        
        
        second_layer_alpha_0j = self.atten(H_0j_cancat).sigmoid()
        
        second_layer_alpha_rj = self.atten(H_rj_concat).sigmoid()
        
       
        
        yX = self.classifierX(HX_second_layer)  # decrease knowledeg to output dimension 
        y_lp = self.classifier_lowpass(H_lp_square)
        y_hp = self.classifier_lowpass(H_hp_square)

        y_0j = y_lp * (1-second_layer_alpha_0j.view(-1, 1)) + yX * second_layer_alpha_0j.view(-1, 1)
        y_rj = y_hp * (1-second_layer_alpha_rj.view(-1, 1)) + yX * second_layer_alpha_rj.view(-1, 1)
        y = y_0j+y_rj
        
        return y
        
        
    def forward(self,low_pass_knowledge,high_pass_knowledge,X): # we can not take list in here, seems i must do the dumb way
        y_reconstruct = self.hidden_layer_imitation(low_pass_knowledge[0],high_pass_knowledge[0], X)
        if self.layers  ==3: 
            y_reconstruct = self.intermediate_node_embedding(low_pass_knowledge[1],high_pass_knowledge[1], y_reconstruct)
        if self.layers >3: 
            y_reconstruct = self.intermediate_node_embedding(low_pass_knowledge[1],high_pass_knowledge[1], y_reconstruct)
            y_reconstruct = self.intermediate_node_embedding(low_pass_knowledge[2],high_pass_knowledge[2], y_reconstruct)
        y = self.output_layer_imtation(low_pass_knowledge[-1],high_pass_knowledge[-1],y_reconstruct)
        return y   

#------------------------------------------FMLPS-----------------------------------
class FMLPS(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layer=1,
                 dropout=.5, norm_type='none'):
        super().__init__()    
        self.mlp_lowpass = nn.Linear(num_nodes, hidden_channels)
        self.mlp_highpass = nn.Linear(num_nodes, hidden_channels)
        self.mlpX = nn.Linear(in_channels, hidden_channels)
        self.atten = nn.Linear(hidden_channels * 2, 1)
        self.classifier_lowpass = MLP(hidden_channels, hidden_channels, out_channels,num_layer, dropout, norm_type=norm_type)
        self.classifier_highpass = MLP(hidden_channels, hidden_channels, out_channels,num_layer, dropout, norm_type=norm_type)
        self.classifierX = MLP(hidden_channels, hidden_channels, out_channels, num_layer,dropout, norm_type=norm_type)
        self.dropout = nn.Dropout(dropout)
        
        
    def simplified_imitation(self, low_pass_square, high_pass_square, X):
        low_pass_square_encoding = self.mlp_lowpass(low_pass_square)
        high_pass_square_encoding = self.mlp_lowpass(high_pass_square)
        X_encoding = self.mlpX(X.double())
        H_0j_cancat = torch.cat((low_pass_square_encoding, X_encoding), dim=1)
        H_rj_concat = torch.cat((high_pass_square_encoding, X_encoding), dim=1)
        
        alpha_0j = self.atten(H_0j_cancat).sigmoid()
        alpha_rj = self.atten(H_rj_concat).sigmoid()
        
        
        
        
        hpmean =np.mean(alpha_rj.detach().numpy())
        print(hpmean)
        # hpmean =np.mean(alpha_rj.detach().numpy())
        # print(hpmean)
        #%store lpmean > 'lpmean.txt
        
        
        
        yX = self.classifierX(X_encoding)  # decrease knowledeg to output dimension 
        y_lp = self.classifier_lowpass(low_pass_square_encoding)
        y_hp = self.classifier_lowpass(high_pass_square_encoding) # expect alpha is smaller when graph is heterophily
        
        y_0j = y_lp * (1-alpha_0j.view(-1, 1)) + yX * alpha_0j.view(-1, 1)
        y_rj = y_hp * (1-alpha_rj.view(-1, 1)) + yX * alpha_rj.view(-1, 1) 
        y = y_0j+y_rj
        
        return y
    
    def forward(self,low_pass_knowledge,high_pass_knowledge,X):
        y = self.simplified_imitation(low_pass_knowledge[-1], high_pass_knowledge[-1], X) # directly apply the highest power to X
        return y  
        









    
    
