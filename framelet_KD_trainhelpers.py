#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:19:42 2023

@author: daishi:
    
    model parameters
"""
import argparse
import torch


"""summary of argements"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)    

#############################    System settings     ################################

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU") # change from cpu to another devices 
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]", # we only need transductive experiment in this paper. 
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )




################################    dataset     ########################################

    """Dataset"""
    parser.add_argument("--dataset", type=str, default='cora',
                        #choices= ['syn_cora-h0.00', 'syn_cora-h0.10', ... ,'syn_cora-h1.00',
                        #Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv,
                        #    'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'film'
                        help='dataset.')
    
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set", # parameter for ogb data 
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set", # parameter for ogb data 
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",# parameter for ogb data 
    )
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.2,  
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.2)
    
    

################Framelet specific: model and optimization#####################
    
    
    """teacher model reated and optimization"""
    parser.add_argument(
        "--num_exp", type=int, default=10, help="Repeat how many experiments"
    )
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,   # lr for cora 
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=1e-2,  # wd for cora 
                        help='weight decay (default: 5e-3)')
    parser.add_argument("--max_epoch", type=int, default=500, help="Evaluate once per how many epochs") 
    parser.add_argument("--patience",type=int,default=50, 
                        help="Early stop is the score on validation set does not improve for how many epochs")
    """model"""
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument("--teacher", type=str, default="spatUFG", help="Teacher model") # spatUFG, simplified_UFG, spatUFG_sdrf,simplified_UFG_sdrf
    parser.add_argument("--student", type=str, default="FMLPO", help="Student model")
    parser.add_argument('--nhid', type=int, default=8, 
                        help='number of hidden units (default: 16)')
    
    parser.add_argument('--epsilon', type=int, default=0, help='epsilon for energy pertubation.')
    parser.add_argument('--Lev', type=int, default=1,
                        help='level of transform (default: 2)')
    parser.add_argument('--scale', type=float, default=2, 
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Linear): Haar, Linear, Quadratic, Sigmoid, Entropy')
    parser.add_argument('--dropout', type=float, default=0,   # drop out for teacher model. set 0 for simplified framelet
                        help='dropout probability (default: 0.7)')
    parser.add_argument('--activation', type=str, default= 'relu',
                        help='activation function (default: relu): None, elu, sigmoid, relu, tanh')
    parser.add_argument('--shrinkage', type=str, default='hard',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='threshold value (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='alpha value in Framelet function (default: 0.5 for Entropy; 20.0 for Sigmoid)')   
    parser.add_argument('--Chebyshev', default=True, action='store_false',
                        help='Whether to use Chebyshev approximation (default: True)')
    parser.add_argument('--FrequencyNum', type=int, default=100,
                        help='The number of (noise) high frequency components (default: 100)')
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )
    parser.add_argument("--batch_size", type=int, default=100000) # change batch size to large if data is relatively small 
    

    """Ablation and student model related"""
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )
    parser.add_argument(
        "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=0,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=8,
        help="Student model hidden layer dimensions",
    )
    parser.add_argument("--dropout_ratio", type=float, default=0.1)  # drop out for student model,0.1 winsconsin
    

    args = parser.parse_args()

    return args


#%%


































    
