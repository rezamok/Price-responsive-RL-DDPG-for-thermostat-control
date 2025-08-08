# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:43:51 2021

@author: kosta
"""
import sys
sys.path.insert(0,'..')
import numpy as np
import os


from kosta.one_for_many import train_second_network
# change working directory
if sys.platform == "win32":
    os.chdir(os.path.abspath(os.path.join(os.path.dirname("__file__"),'..')))
else:
    sys.path.append("../")

import kosta.hyperparams as hp
    
if __name__ == '__main__':
    train_second_network("./kosta/data/splitted-historical-data-no-zero-uk")