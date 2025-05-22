import os
import sys
import re
import random
import yaml
import copy
import time
import argparse
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')
import logging
from logging import getLogger, Logger
from pytz import timezone
from datetime import datetime
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

sys.path.append('./libs')
from load_data import *
from utils import *
from utils_processing import *
from learning import *
from models import *


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument("--config_file", help="config file name")

    parser.add_argument("--exp_name")
    parser.add_argument("--dataset_name")
    parser.add_argument("--main_dir", default="../")
    parser.add_argument("--data_dir", default="../../../opt/nas/data")
    parser.add_argument("--output_dir", default="../output")
    

    parser.add_argument("--weak_cols", type=str, nargs="+", default=['all'], help="list of features to be weak features")


    parser.add_argument("--sample_size", type=int, default=-1, help="all data size. if sample_size = -1, we use all data")
    parser.add_argument("--test_rate", type=float, default=0.5) 
    parser.add_argument("--use_train_size", type=int, default=-1)
    parser.add_argument("--test_size_for_loop", type=int, default=-1)

    parser.add_argument("--measure_time", type=bool, default=True, help="measure each execution times")
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--arch", type=str, default='mlp', help="model architecture")
    parser.add_argument("--hd", type=int, default=500, help='hidden dim for mlp')
    parser.add_argument("--lr", type=float, default=5e-5, help="learning_rate")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    parser.add_argument("--ep", type=int, default=100, help="epochs")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    

    parser.add_argument("--pred_loss", type=str, default='logistic', choices=['logistic', 'log'], help='use loss function for label prediction models')
    parser.add_argument("--est_error_rate", type=float, default=0.5, help='error rate of feature estimation models')

    return parser


def run(args, logger):

    set_seed_torch(args.seed)

    print("data loading...")
    time_s = time.time()

    data_df, cat_cols = load_data(data_name=args.dataset_name,
                              data_path=args.data_dir,
                              sample_size=args.sample_size,
                              seed=args.seed)
    
    
    if args.weak_cols == ['all']:
        weak_cols = cat_cols
    else:
        weak_cols = args.weak_cols

    data_df = exec_ohe(data_df, cat_cols, is_comp=False)

    # train test split
    test_index = sorted(random.sample(data_df.index.tolist(), int(data_df.shape[0] * args.test_rate)))
    train_index = sorted(list(set(data_df.index.tolist()) - set(test_index)))

    ord_train_df = data_df.iloc[train_index].reset_index(drop=True)
    ord_test_df = data_df.iloc[test_index].reset_index(drop=True)

    if args.use_train_size > 0:
        assert args.use_train_size <= len(train_index)
        limited_train_index = sorted(random.sample([i for i in range(len(train_index))], args.use_train_size))
        ord_train_df = ord_train_df.iloc[limited_train_index].reset_index(drop=True)
        
    time_e = time.time()
    logger.info(f"data loading... {time_e - time_s:.1f} [sec]")

    # preparing feature estimation models
    logger.info('preparing feature estimation models ...')
    
    est_train_df = ord_train_df.copy()
    est_test_df = ord_test_df.copy()
    for est_target_name in weak_cols:
        print(f'weak col: {est_target_name}')
        time_s = time.time()
        est_target_ohe_cols = [c for c in list(ord_train_df.columns) if est_target_name in c]
        est_train_df = randomly_estimation(df=est_train_df, est_target_name=est_target_name, error_rate=args.est_error_rate, seed=args.seed + 1000*weak_cols_code(args.dataset_name, [est_target_name]))
        est_test_df = randomly_estimation(df=est_test_df, est_target_name=est_target_name, error_rate=args.est_error_rate, seed=args.seed + 1000*weak_cols_code(args.dataset_name, [est_target_name]))

        train_acc = accuracy_score(y_true=np.argmax(ord_train_df[est_target_ohe_cols].values, axis=1),
                                y_pred=np.argmax(est_train_df[est_target_ohe_cols].values, axis=1))
        test_acc = accuracy_score(y_true=np.argmax(ord_test_df[est_target_ohe_cols].values, axis=1),
                                y_pred=np.argmax(est_test_df[est_target_ohe_cols].values, axis=1))
        
        time_e = time.time()
        logger.info(f"weak_col: {time_e - time_s:.1f}    Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # training label prediction model
    print('training label prediction model ...')
    time_s = time.time()
    
    ## training label prediction model 
    pred_model, pred_ep_table, pred_train_score_dict, pred_test_score_dict \
          = train_pred_model(arch=args.arch, hidden_dim=args.hd,
                            lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=args.device,
                            train_df = est_train_df,
                            test_df = est_test_df,
                            seed=args.seed,
                            loss_func=args.pred_loss,
                            target_name='target',
                            test_size_for_loop=args.test_size_for_loop,
                            )

    time_e = time.time()   
    logger.info(f"training label prediction model ...  {time_e - time_s:.1f} [sec], \n"
                    + f"Test Acc: {pred_test_score_dict['acc']:.4f}, F1: {pred_test_score_dict['f1']:.4f}, AUROC: {pred_test_score_dict['auroc']:.4f} \n"
                    + f"Tr Acc: {pred_ep_table[-1, 1]}, Te Acc: {pred_ep_table[-1, 2]}, Tr Loss: {pred_ep_table[-1, 3]}, Te Loss: {pred_ep_table[-1, 4]}")


    # save log
    save_dict = {}

    save_dict['pred_model (est)'] = pred_model
    save_dict['pred_model_ep_table (est)'] = pred_ep_table
    save_dict['pred_model_train_score (est)'] = pred_train_score_dict
    save_dict['pred_model_test_score (est)'] = pred_test_score_dict


    with open(os.path.join(args.output_dir, args.exp_name, args.log_name, args.log_name +"_log.pkl"), "wb") as f:
        pickle.dump(save_dict, f)





if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args()

    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print(f"Loaded config from {config_file}!")

    if args.device == 'cuda:0':
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    args.task = return_task(args.dataset_name)

    args.exp_name += f"_{args.dataset_name}"

    args.log_name = get_log_filename(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.log_name), exist_ok=True)


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        #filename=os.path.join(args.output_dir, args.exp_name, log_filename),
    )

    logger=getLogger(args.dataset_name)

    # https://qiita.com/r1wtn/items/d615f19e338cbfbfd4d6
    # Set handler to output to files
    fh = logging.FileHandler(os.path.join(args.output_dir, args.exp_name, args.log_name, args.log_name + ".log"))
    fh.setLevel(logging.DEBUG)
    def customTime(*args):
        return datetime.now(timezone('Asia/Tokyo')).timetuple()
    formatter = logging.Formatter(
        fmt='%(levelname)s : %(asctime)s : %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S %z"
    )
    formatter.converter = customTime
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # logging args
    for k, v in config_args.__dict__.items():
        logger.info(f"args[{k}] = {v}")

    run(args, logger)


    


