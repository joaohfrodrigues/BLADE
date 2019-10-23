# -*- coding: utf-8 -*-
"""
Utility functions for the RBM
Created on Fri May 10 2019

@author: João Henrique Rodrigues, IST

version: 1.0
"""
import numpy as np
import pandas as pd
import subprocess
import torch
from torch.utils.data import SubsetRandomSampler
import argparse


def parse_general_file(filepath, n_features, n_time_points, task, input_feat_values=0, ref_col_name=0, sep_symbol=',', label_column=-1):        
    '''Parse a general file to comply to the default format to be used in the
    RBM-tDBN model.

    Parameters
    ----------
    filepath : String
        Path of the file to be parsed.
    n_features: int
        Number of features of the dataset.
    n_time_points: int
        Number of time points in the dataset.
    task: char 
        Task to be performed by the model, learning, classification or 
        prediction. Both classification and prediction require the existence of 
        labels.
    input_feat_values: list, default 0
        Values of the different features present in the dataset.
    ref_col_name : string, default 0
        Name of the reference column, 'subject_id' if possible.
    sep_symbol: char, default ','
        Separation symbol on the file to be parsed.
    label_column: int, default -1
        Column referring to the labels.

    Returns
    -------
    df : dataframe
        Dataframe of the parsed file.
    labels: list
        Different labels present on the dataset.
    feat_values: list
        Set of values taken by the features present in the dataset.
    '''
    if input_feat_values == 0:
        feat_values = []
        label_values = []
        for i in range(n_features):
                feat_values.append([])
    else:
        feat_values=input_feat_values
    
    if ref_col_name == 0:
        df = pd.read_csv(filepath+'.csv', index_col=False, sep=sep_symbol, header=0)
    else:
        df = pd.read_csv(filepath+'.csv', index_col=ref_col_name, sep=sep_symbol,header=0)
    
    df.index.name = 'subject_id'
    
    labels = pd.DataFrame(data=df.values[:,label_column],    # values
                     index=df.index,    # 1st column as index
                     columns=['label'])
    labels.index.name = 'subject_id'
    
    if task == 'c':
        df.rename(columns={df.columns[label_column]: 'label'}, inplace=True)
        labels.index.name = 'subject_id'
        df.drop(columns=['label'], inplace=True)
    
    i=1
    time=0
    for y in range(len(df.columns)):
        df.rename(columns={df.columns[y]: 'X'+str(i)+'__'+str(time)}, inplace=True)
        i+=1
        if i >= n_features+1:
            i=0
            time+=1
        
    i=0
    for x in df:
        for y in range(len(df[x])):
            if input_feat_values == 0:
                if df[x][y] not in feat_values[i]:
                    feat_values[i].append(df[x][y])
                
            df[x][y]=feat_values[i].index(df[x][y])
        
        i+=1
        if i >= n_features:
            i=0
            
    if task == 'c':
        for y in range(len(labels)):
            if labels['label'][y] not in label_values:
                label_values.append(labels['label'][y])
            labels['label'][y] = label_values.index(labels['label'][y])
        labels.to_csv(filepath+'_target.csv',quoting=1)
            
    df.to_csv(filepath+'_parsed.csv',quoting=1)
    
    outF = open(filepath+'_dic.txt', "w")
    
    for i in range(1, n_features+1):
            outF.write('Feature ' + str(i) + ' has ' + str(len(feat_values[i])) + ' different values\n')
            for j in range(len(feat_values[i])):
                outF.write(str(j) + ': ' + str(feat_values[i][j]) + '\n')
    
    if task=='c':
        outF.write('Labels have ' + str(len(label_values)) + ' different values\n')
        for j in range(len(label_values)):
            outF.write(str(j) + ': ' + str(label_values[j]) + '\n')
    
    outF.close()
    
    return df, labels, feat_values    

def create_parser(*args):
    ''' Creates a parser to analyze information given by the user when running
    the program from terminal.
    
    Returns
    ----------
    parser: argparse.ArgumentParser
        Parser which will be use to extract the information.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type')
    parser.add_argument('data_file')
    parser.add_argument('-hu','--hidden_units', type=int, default=50, help='set the number of hidden units')
    parser.add_argument('-bs','--batch_size_ratio', type=float, default = 0.1, help='set the batch size ratio')
    parser.add_argument('-cd','--contrastive_divergence', type=int, default=1, help='set k in cd-k')
    parser.add_argument('-e','--epochs', type=int, default = 30, help='set the number of epochs')
    parser.add_argument('-lr','--learning_rate', type=float, default = 0.05, help='set the learning rate')
    parser.add_argument('-wd','--weight_decay', type=float, default = 1e-4, help='set the weight decay')
    parser.add_argument('-tsr','--test_set_ratio', type=float, default = 0.2, help='set the ratio for the test set')
    parser.add_argument('-vsr','--validation_set_ratio', type=float, default = 0.1, help='set the ratio for the validation set')
    parser.add_argument('-pcd','--persistent_cd', type=bool, default = False, help='activate persistent contrastive divergence')
    parser.add_argument('-v','--version',action='version', version='%(prog)s 2.0')
    parser.add_argument('-vb','--verbose',dest='verbose', default = False, action='store_true',help='enables additional printing')
    parser.add_argument('-nr','--number_runs', type=int, default = 1, help='number of repetitions')
    return parser

def count_labels(filepath):
    ''' Reads the file containing the labels and returns some information.
    
    Parameters
    ----------
    filepath : string
        Path to the label file.
        
    Returns
    ----------
    labels: dataframe
        Dataframe containing all the label information.
    label_values: list
        List with the different label values.
    label_indices:
        Different subject indices corresponding to each label.
    '''
    labels = pd.read_csv(filepath + '_target.csv', index_col= 'subject_id', header=0)
    label_values = []
    label_indices = []
    
    for y in labels.index:
        if labels['label'][y] not in label_values:
            label_values.append(labels['label'][y])
            label_indices.append([])
            label_indices[label_values.index(labels['label'][y])].append(y)
        else:
            label_indices[label_values.index(labels['label'][y])].append(y)
    
    return labels, label_values, label_indices

def parse_file_one_hot(df, n_features, n_time_points, labels=None, n_diff = None):
    ''' Performs one-hot encoding on a dataset.
    
    Parameters
    ----------
    df : dataframe
        Dataframe with the dataset to be encoded.
    n_features: int
        Number of features.
    n_time_points: int
        Number of time points.
    labels: list, default None
        Labels corresponding to each subject.
    n_diff: list, default None
        Different values for each feature.
        
    Returns
    ----------
    labels: dataframe
        Dataframe containing all the label information.
    label_values: list
        List with the different label values.
    label_indices:
        Different subject indices corresponding to each label.
    '''
    if n_diff is None:
        v_max=np.zeros(n_features)
        v_min=np.zeros(n_features)
        
        i=0    
        for x in df:
            if max(df[x]) > v_max[i]:
                v_max[i] = max(df[x])
            i+=1
            if i >= n_features:
                i=0
        
        v_max = v_max.astype(int)
        v_min = v_min.astype(int)
        v_diff = v_max-v_min #different values for the features
        v_diff = v_diff.astype(int)
        n_diff = (v_diff + 1)
    subjects = df.shape[0]
    encoded_data = np.zeros((subjects*n_time_points,sum(n_diff)))
    ext_labels = np.zeros((subjects*n_time_points))
    
    col_aux=0
    time=0
    
    for x in df: #iterate on the features and time
        for y in df.index: #iterate on the subjects
            encoded_data[subjects*time+y-df.index[0]][sum(p for p in n_diff[0:col_aux])+df[x][y].astype(int)]=1
            if labels is not None:
                ext_labels[subjects*time+y-df.index[0]] = labels[y-df.index[0]]
                #training_data[subjects*time_step+y][sum(p for p in n_diff[0:col_aux])+df[x][y]]=1
        col_aux+=1
        if col_aux >= n_features:
            col_aux = 0
            time +=1
    
    return encoded_data, n_diff, ext_labels

def create_train_sets(dataset, label_indices=0, test_train_ratio=0.2, validation_ratio=0.1, batch_size=32, get_indices=True, 
                      random_seed=42, shuffle_dataset=True, label_ratio = False):
    '''Distributes the data into train, validation and test sets and returns the respective data loaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which will be used to train, validate and test the model.
    test_train_ratio : float, default 0.2
        Number from 0 to 1 which indicates the percentage of the data 
        which will be used as a test set. The remaining percentage
        is used in the training and validation sets.
    validation_ratio : float, default 0.1
        Number from 0 to 1 which indicates the percentage of the data
        from the training set which is used for validation purposes.
        A value of 0.0 corresponds to not using validation.
    batch_size : integer, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indices : bool, default True
        If set to True, the function returns the dataloader objects of 
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    random_seed : integer, default 42
        Seed used when shuffling the data.
    shuffle_dataset : bool, default True
        If set to True, the data of which set is shuffled.
    label_indices: array, default 0
        Data indices for the different labels.
    label_ratio: bool, default False
        Whether to maintain or not the each label's ratio when creating the
        sets.

    Returns
    -------
    train_data : torch.Tensor
        Data which will be used during training.
    val_data : torch.Tensor
        Data which will be used to evaluate the model's performance 
        on a validation set during training.
    test_data : torch.Tensor
        Data which will be used to evaluate the model's performance
        on a test set, after finishing the training process.
    '''
    # Create data indices for training and test splits    
    if label_ratio:
        test_split = []
        val_split = []
        for label in range(len(label_indices)):
            test_split.append([])
            val_split.append([])
            test_split[label] = int(np.floor(test_train_ratio * len(label_indices[label])))
            val_split[label] = int(np.floor(validation_ratio * (1-test_train_ratio) * len(label_indices[label])))
        
        if shuffle_dataset:
            #np.random.seed(random_seed)
            for label in range(len(label_indices)):
                np.random.shuffle(label_indices[label])
        
        for label in range(len(label_indices)):
            if label == 0:
                train_indices = label_indices[label][test_split[label]+val_split[label]:]
                val_indices = label_indices[label][test_split[label]:test_split[label]+val_split[label]]
                test_indices = label_indices[label][:test_split[label]]
            else:
                train_indices.extend(label_indices[label][test_split[label]:])
                val_indices.extend(label_indices[label][test_split[label]:test_split[label]+val_split[label]])
                test_indices.extend(label_indices[label][:test_split[label]])
                
        if shuffle_dataset:
            np.random.shuffle(test_indices)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
    else:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_ratio * dataset_size))
        if shuffle_dataset:
            #np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[test_split:], indices[:test_split]
            
        # Create data indices for training and validation splits
        train_dataset_size = len(train_indices)
        val_split = int(np.floor(validation_ratio * train_dataset_size))
        if shuffle_dataset:
            #np.random.seed(random_seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
        train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]

    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    if get_indices:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader, test_dataloader

def weight_analyzer(weights, feat_values, threshold):
    ''' Analyze the weights of the restriced Boltzmann machine
    
    Parameters
    ----------
    weights : list
        Weights learned for the RBM
    feat_values : list
        Values of each feature, used for representation and helping the user
        interpretation.
    threshold: float
        Percentage of the maximum in order to consider that a feature is 
        important for that hidden unit.
    '''
    n_features = len(feat_values)
    print('Units in the same group have higher probability of being active together,'+ 
          'while units in different groups have lower probability of being active together \n')
    max_weight = float(np.absolute(weights).max())
    for j in range(0,weights.shape[1]):
        pos_result = []
        neg_result = []
        #max_weight = max(np.absolute(weights[:,j]))
        for i in range(0,weights.shape[0]):
            if np.absolute(weights[i,j]) > max_weight*threshold:
                if weights[i,j] > 0:
                    pos_result.append(i)
                else:
                    neg_result.append(i)
        
        print('\nH' + str(j))
        print('+')
        
        for i in pos_result:
            print(str(i) + ': X' + str(i%n_features) + '=' + str(feat_values[i%n_features][int(np.floor(i/n_features))]))
        
        print('-')
        for i in neg_result:
            print(str(i) + ': X' + str(i%n_features) + '=' + str(feat_values[i%n_features][int(np.floor(i/n_features))]))          

def jarWrapper(*args):
    ''' Method used to run a Java program from a Python script and get its
    results.
    
    Returns
    ----------
    ret: String
        Results given by the Java program executed.
    '''
    process = subprocess.Popen(['java', '-jar']+list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.decode("utf-8")
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    
    ret += stdout.split('\n')
    if stderr != '':
        ret += stderr.split('\n')
    ret.remove('')
    return ret

def check_int(c):
    try:
        int(c)
        return True
    except ValueError:
        return False
    
def parse_dic(filepath):
    dic = open(filepath+'_dic.txt','r')
    feat_values = []
    
    for line in dic:
        if line[0] == '0':
            line_feat=[]
            feat_values.append(line_feat)
            value = line.split(': ')[1][:-1]
            line_feat.append(value)
        elif check_int(line[0]):
            value = line.split(': ')[1][:-1]
            line_feat.append(value)
        elif line[0] == 'L':
            break
    return feat_values

def reverse_one_hot(data, feat_values):
    if len(data.shape)==2:
        ret=[[] for x in range(data.shape[0])]
        n_instances= data.shape[0]
    else:
        ret=[]
        n_instances= 1
    
    i=0
    for feature in feat_values:
        j=0
        for value in feature:
            if n_instances > 1:
                for k in range(n_instances):
                    if data[k][i*len(feature)+j] == 1:
                        ret[k].append(value)
            else:
                if data[i*len(feature)+j] == 1:
                        ret.append(value)
            j+=1
        i+=1
    
    return ret

def parse_labels(filepath):
    df = pd.read_csv(filepath+'_class.csv', index_col="subject_id", sep=',', header=0)
    write_df = pd.DataFrame(data=df.values,    # values
             index=df.index,    # 1st column as index
             columns=['label'])
    write_df.to_csv(filepath+'_target.csv',quoting=1)