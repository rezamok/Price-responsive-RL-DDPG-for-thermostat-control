# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 13:35:57 2021

@author: kosta
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import kosta.hyperparams as hp

class DatasetARX(Dataset):
    """
    Pytorch like Dataset class for handling ARX model historical data used to 
    train network 2 
    """
    def __init__(self, csv_path):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if not hp.cooling:
            self.df = self.df.loc[self.df.Case>0.5,:]
        if not hp.heating:
            self.df = self.df.loc[self.df.Case<0.5,:]
        state_order = ['T_amb']+[f's{x}' for x in range(1,10)]+\
            ['T_room','Case','Electricity price','Lower bound','Upper bound']
        self.column_ids = []
        for column in state_order:
            column_id = self.df.columns.get_loc(column)
            self.column_ids.append(column_id)
        self.uk_id = self.df.columns.get_loc('uk')
        self.T_room_id = self.df.columns.get_loc('T_room')
        self.T_room_next_id = self.df.columns.get_loc('T_room+1')
        
    def __getitem__(self, i):
        state = self.df.iloc[i,self.column_ids]
        state = np.append(state,0.) # this is for battery
        uk = self.df.iloc[i,self.uk_id]
        delta_t = self.df.iloc[i,self.T_room_next_id]-self.df.iloc[i,self.T_room_id]
        # clip delta_t and scale it to -1, 1
        delta_t = np.clip(delta_t, -hp.max_delta_t, hp.max_delta_t)
        delta_t = delta_t/hp.max_delta_t
        return state, delta_t, uk
    
    def __len__(self):
        return len(self.df)

