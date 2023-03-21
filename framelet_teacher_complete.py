#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:23:06 2023

@author: daishi:


Spatial Framelet and simplified framelet 

teacher model with Haar filter with only one high pass and low pass.     

graph knowledge is adjusted/enhanced by sdrf if the number of layer is high.
    
"""
import time
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import dgl
from dataloader import (load_dataset,load_ogb_data,generate_split,get_pyg_syn_cora)
from scipy.sparse.linalg import lobpcg
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    check_writable,
    compute_min_cut_loss,
)


from scipy import sparse

from torch_geometric.utils import get_laplacian

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from framelet_KD_trainhelpers import get_args
from teacher_student_models import (
    SpatUFG,
    Simplifed_UFG,
    scipy_to_torch_sparse,
    getFilters,
    get_operator,
    get_operator2,
    get_operator1,)

from train_evaluation import(run_transductive,sdrf)
torch.set_default_dtype(torch.float64) 
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#------------------------- load the data and get d_list:-----------------------

args = get_args()



""" Load data """
syn_cora = ['syn_cora-h0.00', 'syn_cora-h0.10','syn_cora-h0.20', 'syn_cora-h0.30','syn_cora-h0.40',
            'syn_cora-h0.50','syn_cora-h0.60','syn_cora-h0.70','syn_cora-h0.80','syn_cora-h0.90','syn_cora-h1.00']
homophily_dataset = ["cora", "citeseer", "pubmed", "computers","photo","cs","physics",'ognb-arxiv']
heterophily_dataset =["chameleon","squirrel","cornell", "texas", "wisconsin","actor"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]

teacher_model = ['spatUFG','simplified_UFG'] # generate orginal dataset. 
teacher_sdrf = ['spatUFG_sdrf','simplified_UFG_sdrf' ] # generate dataset after sdrf 

  
args = get_args()



""" Load data get d_list """

if  args.dataset in syn_cora:
    data_name = args.dataset

    data_name, homolevel = data_name.split('-')
    dataset = get_pyg_syn_cora(".", homolevel, rep=args.num_exp+1)
    data = dataset[0]
    assert data_name == 'syn_cora'
else:
    dataset = load_dataset(args.data_path, args.dataset)        
    data = dataset[0]
    if args.teacher in teacher_sdrf:
        data_sdrf = sdrf(data)
        data_sdrf.x = data.x
        data_sdrf.y = data.y
        data = data_sdrf 
num_classes = dataset.num_classes
num_train = int(len(data.y) / num_classes * args.train_rate)
num_val = int(len(data.y) / num_classes * args.val_rate)
data.train_mask, data.val_mask, data.test_mask = generate_split(data, 
                                                            num_classes, args.seed, 
                                                            num_train, num_val)    

idx_train = torch.where(data.train_mask)
idx_val = torch.where(data.val_mask)
idx_test = torch.where(data.test_mask)
idx_train = torch.cat(idx_train, dim=0)  #shape of the index must be an int
idx_val = torch.cat(idx_val, dim=0)
idx_test = torch.cat(idx_test, dim=0)
edge_index = data["edge_index"].clone().detach()
g = dgl.graph((edge_index[0], edge_index[1]))
g = dgl.to_bidirected(g) 
feats = data.x.clone().detach()
labels = data.y.flatten()    


if args.dataset in OGB_data:
    g, labels, idx_train, idx_val, idx_test = load_ogb_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )

    
args.feat_dim = feats.shape[1]
args.label_dim = labels.int().max().item() + 1
num_nodes= feats.shape[0]

A = g.adjacency_matrix()
edge_index = torch.stack([g.edges()[0],g.edges()[1]])
v = torch.ones(edge_index.shape[1])
i = torch.LongTensor(edge_index)
edge_index, edge_weight = gcn_norm(i, v, feats.shape[0], add_self_loops=True)
symadj = torch.sparse.FloatTensor(edge_index, edge_weight, (feats.shape[0],feats.shape[0])).to(device)
L = get_laplacian(torch.stack((g.edges()[0],g.edges()[1])), num_nodes=num_nodes, normalization='sym')
L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

lobpcg_init = np.random.rand(num_nodes, 1)
lambda_max, _ = lobpcg(L, lobpcg_init)
lambda_max = lambda_max[0]
FrameType = args.FrameType
DFilters = getFilters(FrameType)

# Preparing SVD-Framelet Matrix 
Lev = args.Lev  # level of transform
scale = args.scale  # dilation scale
n = args.n
r = len(DFilters)
    # get matrix operators
if args.Chebyshev:
    if (FrameType == 'Entropy' or FrameType == 'Sigmoid'):
        J = np.log(lambda_max / np.pi) / np.log(scale)
        d = get_operator2(L, DFilters, n, scale, J, Lev)
    else:
        #J = np.log(lambda_max / np.pi) / np.log(scale) + Lev - 1
        J=1   # set J =1 for only one high and low pass domain.
        d = get_operator(L, DFilters, n, scale, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
else:
    lambdas, eigenvecs = np.linalg.eigh(L.todense())
    # lambda_max = lambdas[-1]
    d_list = get_operator1(L, DFilters, lambdas, eigenvecs, scale, Lev)
    d_list = [torch.tensor(x).to(device) for x in d_list]
    



#-------------------------generate the result------------------------------



def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_aug_k > 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        args.teacher = f"GA{args.feature_aug_k}{args.teacher}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    

    args.feat_dim = feats.shape[1]
    args.label_dim = labels.int().max().item() + 1


    """ Model config: we dont need this one for filling in the model """
    conf = {}
    
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")
    print(args.epsilon)
    print("1379:不要回复!",end="")

    """ Model init """   
    if "spatUFG" in conf["teacher"]:    
        model = SpatUFG(args.feat_dim, args.nhid , args.label_dim , g.num_nodes(),
                    d_list, symadj,edge_index,args.num_layers, dropout=args.dropout, epsilon=args.epsilon).to(device)          
                     # optional to set the nhid = label dim for small datasets 
    elif "simplified_UFG" in conf["teacher"]: 
        model = Simplifed_UFG(args.feat_dim, args.label_dim , g.num_nodes(),
                    d_list, symadj,edge_index,args.num_layers, dropout=args.dropout, epsilon=args.epsilon).to(device)
       
    
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(args.dataset)

    """ Data split and run: we directly put transductive and inductive train/eval mehtod here: """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)

        times = []
        t1 =time.time()
        out, score_val, score_test = run_transductive(
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
        )
        score_lst = [score_test]
        
        t2 = time.time()
        times.append(t2-t1)
    """ Saving teacher outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

        
    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)  
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str)
    


if __name__ == "__main__":
    main()
