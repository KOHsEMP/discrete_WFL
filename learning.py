import os
import sys
import re
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
import torcheval.metrics.functional as mF
import torcheval.metrics.classification as mC
from timm.scheduler import CosineLRScheduler

from models import *
from loss_funcs import *
from utils import *

def prepare_dataloader(train_df, test_df, batch_size, target_name='target', dl_generator=None):
    
    # when target is one-hot encoded, target_cols is list, otherwise, target_cols is str (taget_name) 
    target_cols = [c for c in train_df.columns.tolist() if re.search(rf'^{target_name}_', c)]
    if len(target_cols) == 0:
        target_cols = [target_name]
    
    if len(target_cols) == 1: # target col is not one-hot encoded
        target_cols = target_name
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        F.one_hot(torch.tensor(train_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                        )
        test_dataset  = torch.utils.data.TensorDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        F.one_hot(torch.tensor(test_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                        )
    else: # target col is one-hot encoded
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                    torch.tensor(train_df[target_cols].values, dtype=torch.float32))
        test_dataset  = torch.utils.data.TensorDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                    torch.tensor(test_df[target_cols].values, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                                               worker_init_fn=seed_worker, generator=dl_generator)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_df), shuffle=True, 
                                                    worker_init_fn=seed_worker, generator=dl_generator)

    if target_cols == target_name:
        num_classes = len(test_df[target_name].value_counts())
    else:
        num_classes = len(target_cols)

    return full_train_loader, train_loader, test_loader, train_dataset, test_dataset, num_classes

def accuracy_check(loader, model, device):
    soft_max = F.softmax
    total, num_samples = 0, 0
    for instances, labels in loader:
        instances, labels = instances.to(device), labels.to(device)
        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)
        outputs = model(instances)
        sm_outputs = soft_max(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, dim=1)
        total += torch.tensor(predicted == labels, dtype=torch.int8).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples

def loss_check(loader, model, device, loss_func, num_classes):
    total, num_samples = 0, 0
    
    # for label prediction models    
    if loss_func == 'logistic':
        if num_classes ==2:
            criteria = nn.SoftMarginLoss()
        else:
            criteria = nn.MultiMarginLoss()
    elif loss_func == 'log':
        criteria = nn.CrossEntropyLoss() # uses softmax function inside
    
    for instances, labels in loader:
        instances, labels = instances.to(device), labels.to(device)
        outputs = model(instances)

        # for label prediction models    
        if loss_func == 'logistic':
            if num_classes ==2:
                loss = criteria(outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
            else:
                loss = criteria(outputs, torch.argmax(labels, axis=1).to(torch.int64))
        elif loss_func == 'log':
            loss = criteria(outputs, labels)
        # for feature estimation models
        elif loss_func == 'rc':
            loss = rc_loss(outputs, labels)
        elif loss_func == 'cc':
            loss = cc_loss(outputs, labels)
        elif loss_func == 'pc':
            loss = pc_loss(outputs, torch.min(labels, dim=1)[1], num_classes=num_classes) 

        total += loss.detach().cpu() * instances.size(0)
        num_samples += instances.size(0)

    return total / num_samples


def evaluation(loader, model, device, num_classes):
    soft_max = F.softmax

    all_labels = torch.tensor([], dtype=torch.int64)
    all_pred_labels = torch.tensor([], dtype=torch.int64)
    all_pred_probs = torch.tensor([])
    for instances, labels in loader:
        instances, labels = instances.to(device), labels.to(device)
        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)
        outputs = model(instances)
        sm_outputs = soft_max(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, dim=1)

        all_labels = torch.cat([all_labels, labels.detach().cpu().to(torch.int64)])
        all_pred_probs = torch.cat([all_pred_probs, sm_outputs.detach().cpu()], dim=0)
        all_pred_labels = torch.cat([all_pred_labels, predicted.detach().cpu().to(torch.int64)])

    score_dict = {}
    if num_classes == 2:
        score_dict['acc'] = mF.binary_accuracy(input=all_pred_labels, target=all_labels)
        score_dict['f1'] = mF.binary_f1_score(input=all_pred_labels, target=all_labels)
        #score_dict['prec'] = mF.binary_precision(input=all_pred_labels, target=all_labels)
        #score_dict['rec'] = mF.binary_recall(input=all_pred_labels, target=all_labels)
        score_dict['auroc'] = mF.binary_auroc(input=all_pred_labels, target=all_labels)
    else: # multiclass
        score_dict['acc'] = mF.multiclass_accuracy(input=all_pred_probs, target=all_labels, num_classes=num_classes)
        score_dict['f1'] = mF.multiclass_f1_score(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        #score_dict['prec'] = mF.multiclass_precision(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        #score_dict['rec'] = mF.multiclass_recall(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        score_dict['auroc'] = mF.multiclass_auroc(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')

    return score_dict


def train_est_model(arch, lr, bs, ep, wd, device, 
                    est_method, est_target_name, 
                    weak_train_df, weak_test_df, ord_train_df, ord_test_df,
                    hidden_dim=None,
                    seed=42,
                    test_size_for_loop=-1,
                    verbose=False):
    
    dl_generator = set_seed_torch(seed, return_g=True)
    
    full_weak_train_loader, weak_train_loader, weak_test_loader, weak_train_dataset, weak_test_dataset, num_classes \
                = prepare_dataloader(train_df=weak_train_df,
                                    test_df=weak_test_df,
                                    batch_size=bs,
                                    target_name=est_target_name,
                                    dl_generator=dl_generator)

    full_ord_train_loader, ord_train_loader, ord_test_loader, ord_train_dataset, ord_test_dataset, num_classes \
                    = prepare_dataloader(train_df=ord_train_df,
                                        test_df=ord_test_df,
                                        batch_size=bs,
                                        target_name=est_target_name,
                                        dl_generator=dl_generator)
    
    if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
        use_test_index = sorted(random.sample(ord_test_df.index.tolist(), test_size_for_loop))
        loop_full_weak_train_loader, loop_weak_train_loader, loop_weak_test_loader, loop_weak_train_dataset, loop_weak_test_dataset, num_classes \
                        = prepare_dataloader(train_df=weak_train_df,
                                            test_df=weak_test_df.iloc[use_test_index].reset_index(drop=True),
                                            batch_size=bs,
                                            target_name=est_target_name,
                                            dl_generator=dl_generator)
        loop_full_ord_train_loader, loop_ord_train_loader, loop_ord_test_loader, loop_ord_train_dataset, loop_ord_test_dataset, num_classes \
                        = prepare_dataloader(train_df=ord_train_df,
                                            test_df=ord_test_df.iloc[use_test_index].reset_index(drop=True),
                                            batch_size=bs,
                                            target_name=est_target_name,
                                            dl_generator=dl_generator)
    else:
        loop_weak_test_loader = copy.deepcopy(weak_test_loader)
        loop_ord_test_loader = copy.deepcopy(ord_test_loader)

    input_dim = len(weak_train_df.columns) - len([c for c in list(weak_train_df.columns) if est_target_name in c])

    if arch == 'mlp':
        assert hidden_dim is not None
        model = mlp_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=ep, lr_min=1e-6, 
                                warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    
    model.eval()
    with torch.no_grad():
        train_accuracy = accuracy_check(loader=ord_train_loader, model=model, device=device)
        test_accuracy = accuracy_check(loader=loop_ord_test_loader, model=model, device=device)
        train_loss = loss_check(loader=weak_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)
        test_loss = loss_check(loader=loop_weak_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)
    if verbose:
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(0, train_accuracy, test_accuracy, train_loss, test_loss))
    
    save_table = np.zeros(shape=(ep, 5))
    for epoch in range(ep):
        scheduler.step(epoch)
        model.train()
        for i, (instances, labels) in enumerate(weak_train_loader):
            instances, labels = instances.to(device), labels.to(device) # labels is represetated as scaled one-hot encoded partial labels
            optimizer.zero_grad()
            outputs = model(instances)

            if est_method == 'rc':
                loss = rc_loss(outputs, labels)
            elif est_method == 'cc':
                labels[labels != 0] = 1 # enable non-scaled
                loss = cc_loss(outputs, labels)
            elif est_method == 'pc':
                loss = pc_loss(outputs, torch.min(labels, dim=1)[1], num_classes=num_classes) 
            else:
                raise NotImplementedError
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_accuracy = accuracy_check(loader=ord_train_loader, model=model, device=device)
            test_accuracy = accuracy_check(loader=loop_ord_test_loader, model=model, device=device)
            train_loss = loss_check(loader=weak_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)
            test_loss = loss_check(loader=loop_weak_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)
        if verbose:
            print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss
    
    model.eval()
    with torch.no_grad():
        train_score_dict = evaluation(loader=ord_train_loader, model=model, device=device, num_classes=num_classes)
        test_score_dict = evaluation(loader=ord_test_loader, model=model, device=device, num_classes=num_classes)
        train_score_dict['loss'] = loss_check(loader=weak_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)
        test_score_dict['loss'] = loss_check(loader=weak_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes)


    return model, save_table, train_score_dict, test_score_dict


def train_pred_model(arch, lr, bs, ep, wd, device, 
                    train_df, test_df,
                    hidden_dim=None,
                    loss_func='logistic',
                    seed=42,
                    target_name='target',
                    test_size_for_loop=-1,
                    verbose=False):
    
    dl_generator = set_seed_torch(seed, return_g=True)
    
    full_train_loader, train_loader, test_loader, train_dataset, test_dataset, num_classes \
                = prepare_dataloader(train_df=train_df,
                                    test_df=test_df,
                                    batch_size=bs,
                                    target_name=target_name,
                                    dl_generator=dl_generator,
                                    )
    
    if test_size_for_loop > 0 and test_size_for_loop < test_df.shape[0]:
        use_test_index = sorted(random.sample(test_df.index.tolist(), test_size_for_loop))
        loop_full_train_loader, loop_train_loader, loop_test_loader, loop_train_dataset, loop_test_dataset, num_classes \
                = prepare_dataloader(train_df=train_df,
                                    test_df=test_df.iloc[use_test_index].reset_index(drop=True),
                                    batch_size=bs,
                                    target_name=target_name,
                                    dl_generator=dl_generator,
                                    )
    else:
        loop_test_loader = copy.deepcopy(test_loader)
    
    input_dim = len(train_df.columns) - 1 
    
    model = mlp_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=ep, lr_min=1e-6, 
                                warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    
    if loss_func == 'logistic':
        if num_classes ==2:
            criteria = nn.SoftMarginLoss()
        else:
            criteria = nn.MultiMarginLoss()
    elif loss_func == 'log':
        criteria = nn.CrossEntropyLoss() # uses softmax function inside
    else:
        raise NotImplementedError

    model.eval()
    with torch.no_grad():
        train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
        test_accuracy = accuracy_check(loader=loop_test_loader, model=model, device=device)
        train_loss = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        test_loss = loss_check(loader=loop_test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
    if verbose:
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(0, train_accuracy, test_accuracy, train_loss, test_loss))
    
    save_table = np.zeros(shape=(ep, 5))
    for epoch in range(ep):
        scheduler.step(epoch)
        model.train()
        for i, (instances, labels) in enumerate(train_loader):
            instances, labels = instances.to(device), labels.to(device) # labels is represetated as scaled one-hot encoded partial labels
            
            optimizer.zero_grad()
            outputs = model(instances)

            if loss_func == 'logistic':
                if num_classes == 2:
                    loss = criteria(outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
                else:
                    loss = criteria(outputs, torch.argmax(labels, axis=1).to(torch.int64))
            elif loss_func == 'log':
                loss = criteria(outputs, labels)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
            test_accuracy = accuracy_check(loader=loop_test_loader, model=model, device=device)
            train_loss = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
            test_loss = loss_check(loader=loop_test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        if verbose:
            print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss

    model.eval()
    with torch.no_grad():
        train_score_dict = evaluation(loader=train_loader, model=model, device=device, num_classes=num_classes)
        test_score_dict = evaluation(loader=test_loader, model=model, device=device, num_classes=num_classes)
        train_score_dict['loss'] = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        test_score_dict['loss'] = loss_check(loader=test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)

    return model, save_table, train_score_dict, test_score_dict


