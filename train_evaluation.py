#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:07:21 2023

@author: daishi

Train and evaluation scheme for both teacher and student models

also include:  computation of Blanced forman curvature, stochastic forman ricci flow. 


"""
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

#%% teacher model train, evaluation and run  without minibatch
def train(model, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    here for framelet we need input as x and d_list 
    """
    model.train()

    # Compute loss and prediction
    logits = model(feats)  
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def evaluate(model ,feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        logits = model(feats)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss.item(), score


def run_transductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)


    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train(model,feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0: 
            out, loss_train, score_train = evaluate(
                model, feats, labels, criterion, evaluator, idx_train
                )
        # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
            loss_val = criterion(out[idx_val], labels[idx_val]).item()
            score_val = evaluator(out[idx_val], labels[idx_val])
            loss_test = criterion(out[idx_test], labels[idx_test]).item()
            score_test = evaluator(out[idx_test], labels[idx_test])
        
            logger.debug(
            f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
        )
            loss_and_score += [[epoch,loss_train, loss_val,loss_test, score_train,score_val, score_test,]]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break


    model.load_state_dict(state)
    
    out, _, score_val = evaluate(
            model ,feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test





##                  ''' student model train,evaluation and run '''




def FMLPO_distill_train(model, low_pass_knowledge, high_pass_knowledge, feats, labels, 
                        criterion, optimizer, lamb=1):
   
    model.train()
    total_loss = 0
    logits = model(low_pass_knowledge,high_pass_knowledge,feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out, labels)
    total_loss += loss.item()
    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return total_loss     







def FMLPO_distill_evaluate(
    model, low_pass_knowledge, high_pass_knowledge,
    feats, labels, criterion,  evaluator, idx_eval=None
):
    """
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        out_list = []
        logits = model(low_pass_knowledge,
                        high_pass_knowledge,
                        feats)
        out = logits.log_softmax(dim=1)
        out_list += [out.detach()]
        out_all = torch.cat(out_list)
        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score






def framelet_distill_run_transductive(  # Run MLPO without using mini batch 
    conf,
    model,
    low_pass_knowledge,
    high_pass_knowledge,
    num_layers,
    feats,
    labels,
    out_t_all, 
    distill_indices, 
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    

    out_t: Soft labels produced by the teacher model.
    criterion_l (nn.NLLLoss())  & criterion_t (KL divergence): 
    
    Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    #batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices 
    #idx_train, idx_val, idx_test = indices 

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    

        
    """assign index to feats, labels and knowledge"""
    
    feats_l,feats_t,feats_val,feats_test = feats[idx_l],feats[idx_t],feats[idx_val],feats[idx_test]
            
    labels_l,out_t,labels_val,labels_test = labels[idx_l],out_t_all[idx_t],labels[idx_val],labels[idx_test]
    
    low_pass_l = [] 
    low_pass_t = []
    low_pass_val = [] 
    low_pass_test = []
    high_pass_l = []
    high_pass_t = []
    high_pass_val = []
    high_pass_test = []
    for i in range(len(low_pass_knowledge)):
        low_pass_l.append(low_pass_knowledge[i][idx_l])
        low_pass_t.append(low_pass_knowledge[i][idx_t])
        low_pass_val.append(low_pass_knowledge[i][idx_val])
        low_pass_test.append(low_pass_knowledge[i][idx_test])
        high_pass_l.append(high_pass_knowledge[i][idx_l])
        high_pass_t.append(high_pass_knowledge[i][idx_t])
        high_pass_val.append(high_pass_knowledge[i][idx_val])
        high_pass_test.append( high_pass_knowledge[i][idx_test])

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["epochs"] + 1):

        loss_l = FMLPO_distill_train(
            model, low_pass_l,high_pass_l,
            feats_l, 
            labels_l,  criterion_l, 
            optimizer,lamb
        )
        loss_t = FMLPO_distill_train(
            model, low_pass_t, high_pass_t,
            
            feats_t, out_t, 
            criterion_t, optimizer,1-lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = FMLPO_distill_evaluate(
                model, low_pass_l, high_pass_l,
                
                feats_l, labels_l, criterion_l , evaluator
            )
            _, loss_val, score_val = FMLPO_distill_evaluate(
                model, low_pass_val, high_pass_val,
                
                 feats_val, labels_val, criterion_l, evaluator
            )
            _, loss_test, score_test = FMLPO_distill_evaluate(
                model, low_pass_test, high_pass_test,
                
                feats_test, labels_test, criterion_l, evaluator
            )

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
        
        model.load_state_dict(state)
    out, _, score_val = FMLPO_distill_evaluate(
        model, low_pass_knowledge, high_pass_knowledge, feats, labels, criterion_l, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test
        


#---------------------------FMLPS train and evaluate with mini-batch, optionally to apply the same change to MLPO 

def FMLPS_distill_train_mini_batch(model, 
                             low_pass_square, high_pass_square, 
                             feats, labels,  criterion,batch_size, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. 
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, int(feats.shape[0] / batch_size))
    #num_batches = int(feats.shape[0]) / int(batch_size)
    
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # 
        logits = model(
                       low_pass_square[idx_batch[i]],high_pass_square[idx_batch[i]],
                       feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def FMLPS_distill_evaluate_mini_batch(
    model, 
    low_pass_square,high_pass_square, 
    feats, labels, criterion, 
    batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            logits = model(
                            low_pass_square[batch_size * i : batch_size * (i + 1)],
                            high_pass_square[batch_size * i : batch_size * (i + 1)],
                            feats[batch_size * i : batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])
        

    return out_all, loss.item(), score


def FMLPS_distill_minibatch_transductive(  # FMLPS transductive with mini batch
    conf,
    model,
    low_pass_square,
    high_pass_square,
    feats,
    labels,
    out_t_all, 
    distill_indices, 
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices 
    #idx_train, idx_val, idx_test = indices 

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)   
    
    
    feats_l, labels_l, low_pass_square_l, high_pass_square_l = feats[idx_l], labels[idx_l],low_pass_square[idx_l],high_pass_square[idx_l]
    
    feats_t, out_t, low_pass_square_t, high_pass_square_t = feats[idx_t], out_t_all[idx_t], low_pass_square[idx_t],high_pass_square[idx_t]
    
    feats_val, labels_val, low_pass_square_val, high_pass_square_val = feats[idx_val], labels[idx_val],low_pass_square[idx_val], high_pass_square[idx_val]
      
    feats_test, labels_test,low_pass_square_test,high_pass_square_test = feats[idx_test], labels[idx_test],low_pass_square[idx_test],high_pass_square[idx_test]        
    
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = FMLPS_distill_train_mini_batch(
            model, 
            low_pass_square_l,high_pass_square_l,
            feats_l, 
            labels_l,  criterion_l, batch_size,
            optimizer,lamb
        )
        
        loss_t = FMLPS_distill_train_mini_batch(
            model, 
            low_pass_square_t,high_pass_square_t, 
            feats_t, out_t, 
            criterion_t, batch_size, 
            optimizer,1-lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = FMLPS_distill_evaluate_mini_batch(
                model, 
                low_pass_square_l,
                high_pass_square_l,
                feats_l, labels_l, criterion_l ,batch_size, evaluator
            )
            _, loss_val, score_val = FMLPS_distill_evaluate_mini_batch(
                model, 
                low_pass_square_val,high_pass_square_val,
                 feats_val, labels_val, criterion_l, batch_size,evaluator
            )
            _, loss_test, score_test = FMLPS_distill_evaluate_mini_batch(
                model, 
                low_pass_square_test,high_pass_square_test,    
                feats_test, labels_test, criterion_l, batch_size, evaluator
            )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
        
        model.load_state_dict(state)
    out, _, score_val = FMLPS_distill_evaluate_mini_batch(
        model,  low_pass_square,
            high_pass_square,feats, labels, criterion_l,batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test








#---------------------------------------------stochastic ricci flow --------------------------------------------    


def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0:
                C[i, j] = 0
                break

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                break

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)

def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C

@jit(nopython=True)
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    for I in prange(dim_i):
        for J in prange(dim_j):
            i = i_neighbors[I]
            j = j_neighbors[J]

            if (i == j) or (A[i, j] != 0):
                D[I, J] = -1000
                break

            # Difference in degree terms
            if j == x:
                d_in_x += 1
            elif i == y:
                d_out_y += 1

            if d_in_x * d_out_y == 0:
                D[I, J] = 0
                break

            if d_in_x > d_out_y:
                d_max = d_in_x
                d_min = d_out_y
            else:
                d_max = d_out_y
                d_min = d_in_x

            # Difference in triangles term
            A2_x_y = A2[x, y]
            if (x == i) and (A[j, y] != 0):
                A2_x_y += A[j, y]
            elif (y == j) and (A[x, i] != 0):
                A2_x_y += A[x, i]

            # Difference in four-cycles term
            sharp_ij = 0
            lambda_ij = 0
            for z in range(N):
                A_z_y = A[z, y] + 0
                A_x_z = A[x, z] + 0
                A2_z_y = A2[z, y] + 0
                A2_x_z = A2[x, z] + 0

                if (z == i) and (y == j):
                    A_z_y += 1
                if (x == i) and (z == j):
                    A_x_z += 1
                if (z == i) and (A[j, y] != 0):
                    A2_z_y += A[j, y]
                if (x == i) and (A[j, z] != 0):
                    A2_x_z += A[j, z]
                if (y == j) and (A[z, i] != 0):
                    A2_z_y += A[z, i]
                if (z == j) and (A[x, i] != 0):
                    A2_x_z += A[x, i]

                TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            D[I, J] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
            )
            if lambda_ij > 0:
                D[I, J] += sharp_ij / (d_max * lambda_ij)



def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = np.zeros((len(i_neighbors), len(j_neighbors)))

    _balanced_forman_post_delta(
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D




def sdrf(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
):
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    if is_undirected:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    else:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = 1.0
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    C = np.zeros((N, N))

    for x in range(loops):
        can_add = True
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin()
        x = ix_min // N
        y = ix_min % N

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break

    return from_networkx(G)














   











