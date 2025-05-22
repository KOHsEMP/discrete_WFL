import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import requests
import pickle
import arff
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from load_data import *


def set_seed(seed=42):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 

def set_seed_torch(seed=42, return_g=False):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    if return_g:
        return g

# https://qiita.com/north_redwing/items/1e153139125d37829d2d
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def show_score_name(score_name):
    show_score_dict = {
        'accuracy_score': 'Accuracy',
        'f1_score': 'F1',
        'f1_score_macro': 'F1 (macro)',
        'recall_score': 'Recall',
        'recall_score_macro': 'Recall (macro)',
        'precision_score': 'Precision',
        'precision_score_macro': 'Precision (macro)',
        'cross_entropy': 'Cross Entropy',
        'entropy': 'Entropy'
    }

def show_score_abbrev(score_name):
    show_score_dict = {
        'accuracy_score': 'Acc',
        'f1_score': 'F1',
        'f1_score_macro': 'F1',
        'recall_score': 'Rec',
        'recall_score_macro': 'Rec',
        'precision_score': 'Prec',
        'precision_score_macro': 'Prec',
        'cross_entropy': 'CE',
        'entropy': 'SE'
    }

    return show_score_dict[score_name]

def show_dataset_name(dataset_name):
    show_dataset_dict = {
        'bank':'Bank',
        'adult': 'Adult',
        'diabetes': 'Diabetes',
        'default': 'Default',
    }

    return show_dataset_dict[dataset_name]


def return_task(dataset_name):

    if dataset_name in ['bank', 'adult', 'diabetes', 'default', 'kick', 'census']:
        return 'classification'
    else:
        return 'regression'
    

def get_log_filename(args):
    
    name = ""

    name += args.dataset_name
    
    if 'weaken_mode' in args.__dict__.keys():
        name += '_W' + args.weaken_mode
    
    if 'weak_cols' in args.__dict__.keys():
        if args.weak_cols == ['all']:
            name += '_WcA'
        else:
            name += '_Wc' + str(weak_cols_code(args.dataset_name, args.weak_cols))

    if 'weaken_mode' in args.__dict__.keys():
        if args.weaken_mode == 'partial':
            name += '_Wn' + str(args.partial_num)
    
    name += '_size' + str(args.sample_size)
    name += '_test' + str(args.test_rate)
    name += '_train' + str(args.use_train_size)

    name += '_L' + args.pred_loss

    if 'est_method' in args.__dict__.keys():
        name += '_estM' + args.est_method
    
    if 'est_error_rate' in args.__dict__.keys():
        name += '_er' + str(args.est_error_rate)

    if 'lam' in args.__dict__.keys():
        name += '_lam' + str(args.lam)

    name += '_' + str(args.arch) 
    if args.arch == 'mlp':
        name += '_hd' + str(args.hd)
    name += '_lr' + str(args.lr)
    name += '_bs' + str(args.bs)
    name += '_ep' + str(args.ep)
    name += '_wd' + str(args.wd)

    name += '_seed' + str(args.seed)

    return name


def pseudo_args(args_dict):
    class Args():
        tmp = "ttt"
    args = Args()
    for k, v in args_dict.items():
        if v is not None:
            setattr(args, k, v)
    return args