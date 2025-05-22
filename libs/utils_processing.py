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

from utils import *
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
def exec_ohe(df, ohe_cols, is_comp=False):
    ohe = OneHotEncoder(sparse_output=False, categories='auto')
    ohe.fit(df[ohe_cols])

    tmp_columns = []
    for i, col in enumerate(ohe_cols):
        tmp_columns += [f'{col}_{v}' for v in ohe.categories_[i]]
    
    df_tmp = pd.DataFrame(ohe.transform(df[ohe_cols]), columns=tmp_columns)
    
    # if the features are represented as complementary label, the value of the index assigned to the complementary label should be 0
    if is_comp: 
        df_tmp = df_tmp * (-1) + 1
    output_df = pd.concat([df.drop(ohe_cols, axis=1), df_tmp], axis=1)

    return output_df


def privacy_transform(df, mask_feature_list, mode="comp", seed=42):
    '''
    Transforming OF to CF due to privacy concerns
    Args:
        df: pd.DataFrame
            original data (not one-hot encoded)
        mask_feature_list: list
            list of feature names to be complementary transformed
        mode: str
            comp:complementary, partial:partial
    Return
        transformed_df: pd.DataFrame 
    '''


    set_seed(seed)
    output_df = df.copy()

    if mode == "comp":
        for comp_feature_name in mask_feature_list:
            val_list = output_df[comp_feature_name].unique().tolist() # list of unique values of the column
        
            # select complementary value uniformly
            output_df[comp_feature_name] = output_df[comp_feature_name].map(lambda x: random.choice(sorted(list(set(val_list) - set([x])))) )
            #for i in range(output_df.shape[0]):
            #    output_df[comp_feature_name].iloc[i] = random.choice(sorted(list(set(val_list) - set([output_df[comp_feature_name].iloc[i]]))))
    
    elif mode == "partial":
        raise NotImplementedError
    

    return output_df

def replace_zeros_with_ones(arr, num_replace):

    zero_indices = np.argwhere(arr == 0).reshape(1,-1)[0]
    
    if len(zero_indices) < num_replace:
        raise ValueError("num_replace is larger than the number of zero elements in the array.")
    
    random_indices = np.random.choice(zero_indices, size=num_replace, replace=False)
    
    #for idx in random_indices:
    #    arr[zero_indices[idx]] = 1
    arr[random_indices] = 1
    
    return arr

def weaken_categorical(df, weak_feature_list, mode, seed=42, 
                       normalization=True,
                       partial_num=None, partial_num_dict=None, 
                       return_comp_as_partial=True):
    '''
    Args:
        df: pd.DataFrame
            categorical features are not one-hot encoded
        weak_feature_list: 
            feature name list to be weakened
        mode: str
            'comp', 'partial'
        return_comp_as_partia: bool
            True -> returend as partial label (including ground-truth), 
            False -> returned as complementry label (not including ground-truth)
    Return:
        df: pd.DataFrame
            weakened featureas are one-hot encoded
    '''
    set_seed(seed)
    output_df = df.copy()

    if mode == 'comp':
        for weak_feature_name in weak_feature_list:
            val_list = output_df[weak_feature_name].unique().tolist() # list of unique values of the column
            # select complementary value uniformly
            output_df[weak_feature_name] = output_df[weak_feature_name].map(lambda x: random.choice(sorted(list(set(val_list) - set([x])))) )

        output_df = exec_ohe(output_df, weak_feature_list, is_comp=return_comp_as_partial)

        if return_comp_as_partial:
            if normalization:
                for col in weak_feature_list:
                    col_ohe_idx = [i for i, c in enumerate(output_df.columns.tolist()) if col in c]
                    output_df.iloc[:, col_ohe_idx] *= 1.0/(len(col_ohe_idx) -1) 


    elif mode == 'partial':
        assert partial_num is not None or partial_num_dict is not None

        output_df = exec_ohe(output_df, weak_feature_list, is_comp=False)

        if partial_num is not None:
            partial_num_dict = {}
            for weak_feature_name in weak_feature_list:
                n_uniqs = len([col for col in output_df.columns if weak_feature_name in col])
                partial_num_dict[weak_feature_name] = np.min([partial_num, n_uniqs-2]) 

        
        for weak_feature_name in weak_feature_list:
            ohe_name_list = [col for col in output_df.columns if weak_feature_name in col]
            output_df[ohe_name_list] = output_df[ohe_name_list].apply(lambda row: replace_zeros_with_ones(row, partial_num_dict[weak_feature_name]), axis=1)

        if normalization:
            for col in weak_feature_list:
                col_ohe_idx = [i for i, c in enumerate(output_df.columns.tolist()) if col in c]
                output_df.iloc[:, col_ohe_idx] *= 1.0/(partial_num_dict[col] +1) 

    return output_df

def randomly_estimation(df, est_target_name, error_rate, seed):
    '''
    Args:
        df: pd.DataFrame
            est_target is one-hot encoded
    '''

    set_seed(seed)
    output_df = df.copy()

    est_target_ohe_cols = [c for c in list(output_df.columns) if est_target_name in c]
    val_list = [v for v in range(len(est_target_ohe_cols))]

    # extract error indices
    error_index = sorted(random.sample(output_df.index.tolist(), int(output_df.shape[0] * error_rate)))
    ## cancel one-hot encoding
    error_est_target_vals = output_df.loc[error_index, est_target_ohe_cols].values
    error_est_target_vals = np.argmax(error_est_target_vals, axis=1) 
    ## insert error
    error_df = pd.Series(error_est_target_vals)
    error_df = error_df.map(lambda x: random.choice(sorted(list(set(val_list) - set([x])))))
    ## re one-hot encoding
    #error_df = pd.DataFrame(np.identity(len(val_list))[error_df.values], columns=est_target_ohe_cols) 
    output_df.loc[error_index, est_target_ohe_cols] = np.identity(len(val_list))[error_df.values]

    return output_df


def hard_labeling(df, comp_cols):
    output_df = df.copy()
    N = df.shape[0]

    for col in comp_cols:
        col_onehot_list = [c for c in df.columns.tolist() if col in c]
        soft_labels = df.loc[:, col_onehot_list].values
        hard_labels = np.zeros(soft_labels.shape, dtype=np.float32)

        argmax_list = np.argmax(soft_labels, axis=1).tolist()

        for i in range(N):
            hard_labels[i, argmax_list[i]] = 1
    
        output_df.loc[:, col_onehot_list] = hard_labels
    
    return output_df