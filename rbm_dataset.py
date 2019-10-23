# -*- coding: utf-8 -*-
"""
Dataset class to be used in the RBM-tDBN model
Created on Wed Apr 17 2019

@author: João Henrique Rodrigues, IST

version: 1.0
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from rbm_utils import parse_file_one_hot
import warnings
warnings.filterwarnings("ignore") #ignore warnings

class RBMDataset(Dataset):
    ''' Dataset class to be used when running the RBM-tDBN model
    '''
    def __init__(self, data_frame, n_features, n_time_points,labels, n_diff = None):
        '''Initialization method for the class.
        
        Parameters
        ----------
        data_frame : DataFrame
            DataFrame with data representation.
        n_features: int
            Number of features.
        n_time_points : int
            Number of time points.
        labels : list
            Labels of the given instances on the dataframe.
        n_diff: list, default None
            Different values of the features.
        '''
        self.data_frame = data_frame
        if n_diff is None:
            self.data, self.n_diff, self.labels = parse_file_one_hot(self.data_frame,n_features, n_time_points, labels)
        else:
            self.n_diff = n_diff
            self.data, _, self.labels = parse_file_one_hot(self.data_frame,n_features, n_time_points, labels, n_diff)
        self.ext_instances = len(self.data)

    def __len__(self):
        return self.ext_instances

    def __getitem__(self, idx):
        value = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(np.array(self.labels[idx])).float()

        return value, label