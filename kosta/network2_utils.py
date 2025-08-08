# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:39:19 2021

@author: kosta
"""
import torch 
import numpy as np
import pandas as pd
import os
from kosta.dataset import DatasetARX
from torch.utils.data import DataLoader
import torch.nn as nn
import kosta.hyperparams as hp

def get_actual_delta_t(batch_state, batch_next_state):
    """
    From state and next_state get the actual temperature change, because
    states does NOT contain only the room temperature.
    """
    batch_delta_t = torch.stack(
        [y[10]-x[10] for x,y in zip(batch_state,batch_next_state)]
        ).reshape(-1,1)
    # batch_delta_t = torch.clip(batch_delta_t,-hp.max_delta_t,hp.max_delta_t)
    return batch_delta_t / hp.max_delta_t
def get_batches_from_experiences(experiences):
    """
    Get batches of state, next_state, action, reward, delta_t from experiences
    sampled from replay buffer.
    """
    # print("mina")
    # print(experiences[0])
    batch_state = torch.stack([x[0]['state'] for x in experiences])
    batch_next_state = torch.stack([x[0]['next_state'] for x in experiences])
    batch_action = torch.stack([x[0]['real_action'] for x in experiences])
    batch_delta_t = torch.stack([x[0]['action'] for x in experiences])
    batch_reward = np.stack([x[0]['reward'] for x in experiences])
    return batch_state,batch_next_state,batch_reward,batch_action,batch_delta_t

def update_network2(batch_state,batch_next_state,batch_action,network2,net2_optim,net2_loss_fcn,batch_delta_t=None):
    """
    Update weights of network 2
    """
    if batch_delta_t is None:
        batch_delta_t = get_actual_delta_t(batch_state, batch_next_state)
    batch_delta_t = batch_delta_t.reshape(-1,1)
    # batch_pred = network2(torch.cat([batch_delta_t, batch_state],dim=-1))
    batch_pred = batch_delta_t
    
    # loss = net2_loss_fcn(batch_pred, batch_action)
    # net2_optim.zero_grad()
    # loss.backward()
    # net2_optim.step()
    loss = 0
    return loss

def network2_train_step(train_loader,network2,optim,loss_fcn,device):
    """
    Train step for training network 2 in algorithm 2. 
    It is used by train_second_network function in one_for_many.py
    """
    network2.train()
    losses = []
    for i,(state,delta_t,action) in enumerate(train_loader):
        state = state.float().to(device)
        delta_t = delta_t.float().to(device)
        action = action.float().to(device).reshape(-1,1)
        loss = update_network2(
            batch_state=state,
            batch_next_state=None,
            batch_action=action,
            network2=network2,
            net2_optim=optim,
            net2_loss_fcn=loss_fcn,
            batch_delta_t=delta_t
        )
        losses.append(float(loss.detach().cpu().numpy()))
        print(f'\riter {i}/{len(train_loader)}',end='')
    print('')
    return losses

def network2_val_step(val_loader,network2,loss_fcn,device):
    """
    Val step for training network 2 in algorithm 2. 
    It is used by train_second_network function in one_for_many.py
    """
    network2.eval()
    #add by mina
   # network2.to(device)
    #loss_fcn = loss_fcn.to(device)
    losses = []
    preds = []
    actions = []
    for i,(state,delta_t,action) in enumerate(val_loader):
        state = state.float().to(device)
        delta_t = delta_t.float().to(device).reshape(-1,1)
        action = action.float().to(device).reshape(-1,1)
        with torch.no_grad():
            pred = network2(torch.cat([delta_t,state],dim=-1))
            loss = loss_fcn(pred,action)
        #     preds.append(pred.detach().cpu().numpy())
        #     actions.append(action.detach().cpu().numpy())
        # losses.append(float(loss.detach().cpu().numpy()))
            preds.append(pred.detach().cpu().numpy())
            actions.append(action.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu().numpy()))
        print(f'\riter {i}/{len(val_loader)}',end='')
    print('')
    preds = np.concatenate(preds,axis=0)
    actions = np.concatenate(actions,axis=0)
    return losses, preds, actions

def get_dataloaders(split_path, batch_size):
    """
    Get data loaders for network 2
    
    Parameters
    -----
    split_path : path to splitted dataset for some rooom (folder that contains
        train.csv,val.csv and test.csv)
    batch_size : 
    """
    train_dataset = DatasetARX(os.path.join(split_path,'train.csv'))
    val_dataset = DatasetARX(os.path.join(split_path,'val.csv'))
    test_dataset = DatasetARX(os.path.join(split_path,'test.csv'))
    # define data loaders
    train_loader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size,shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size,shuffle=False,drop_last=True)
    return train_loader, val_loader, test_loader
def get_loss_fcn(fcn_type):
    if fcn_type == 'mse':
        loss_fcn = nn.MSELoss()
    elif fcn_type == 'mae':
        loss_fcn = nn.L1Loss()
    else:
        raise ValueError(f'Loss function {fcn_type} is not implemented')
    return loss_fcn
