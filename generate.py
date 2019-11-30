# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:26:50 2019

@author: joaor
"""
import numpy as np
import pandas as pd

n_instances = 400
n_time_points = 5

def generate_binomial_1(n_instances,n_time_points):
    n_features=2

    data = np.zeros([n_instances, n_features*n_time_points])
    data[:,0] = np.random.binomial(1, 0.5, n_instances)
    
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            if data[i,0] == 0:
                data[i,1] = np.random.binomial(1, 0.1, 1)
            else:
                data[i,1] = np.random.binomial(1, 0.9, 1)
            
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0 and data[i,t*n_features+1] == 0:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features] == 1 and data[i,t*n_features+1] == 1:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.5, 1)
            
        #LABEL 1
        elif labels[i] == 1:
            if data[i,0] == 0:
                data[i,1] = np.random.binomial(1, 0.1, 1)
            else:
                data[i,1] = np.random.binomial(1, 0.9, 1)
            
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0 and data[i,t*n_features+1] == 0:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.9, 1)
                elif data[i,t*n_features] == 1 and data[i,t*n_features+1] == 1:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+2] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+3] = np.random.binomial(1, 0.5, 1)

    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
    
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    df.index.name = 'subject_id'    
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('binomial_1_'+str(n_time_points)+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('binomial_1_'+str(n_time_points)+'_target.csv',quoting=1)

def generate_binomial_2(n_instances,n_time_points):
    n_features=5

    data = np.zeros([n_instances, n_features*n_time_points])
    data[:,0] = np.random.binomial(1, 0.5, n_instances)
    data[:,1] = np.random.binomial(1, 0.5, n_instances)
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            if data[i,1] == 0:
                data[i,2] = np.random.binomial(1, 0.9, 1)
                data[i,3] = np.random.binomial(1, 0.1, 1)
            else:
                data[i,2] = np.random.binomial(1, 0.1, 1)
                data[i,3] = np.random.binomial(1, 0.9, 1)
                
            if data[i,2] == 0 and data[i,3] == 1:
                data[i,4] = np.random.binomial(1, 0.1, 1)
            elif data[i,2] == 1 and data[i,3] == 0:
                data[i,4] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,4] = np.random.binomial(1, 0.5, 1)
                
                
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.7, 1)
                else:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.3, 1)
                    
                if data[i,t*n_features+5] == 0:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.9, 1)
                
                if data[i,t*n_features+6] == 0:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+8] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+8] = np.random.binomial(1, 0.9, 1)
                    
                if data[i,t*n_features+7] == 0 and data[i,t*n_features+8] == 1:
                    data[i,t*n_features+9] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features+7] == 1 and data[i,t*n_features+8] == 0:
                    data[i,t*n_features+9] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+9] = np.random.binomial(1, 0.5, 1)
        #LABEL 1
        elif labels[i] == 1:
            if data[i,1] == 0:
                data[i,2] = np.random.binomial(1, 0.1, 1)
                data[i,4] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,2] = np.random.binomial(1, 0.9, 1)
                data[i,4] = np.random.binomial(1, 0.1, 1)
                
            if data[i,2] == 1 and data[i,4] == 0:
                data[i,3] = np.random.binomial(1, 0.1, 1)
            elif data[i,2] == 0 and data[i,4] == 1:
                data[i,3] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,3] = np.random.binomial(1, 0.5, 1)
                
                
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.3, 1)
                else:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.7, 1)
                
                if data[i,t*n_features+5] == 0:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.9, 1)
                
                if data[i,t*n_features+6] == 0:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+9] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+9] = np.random.binomial(1, 0.1, 1)
                    
                if data[i,t*n_features+7] == 1 and data[i,t*n_features+9] == 0:
                    data[i,t*n_features+8] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features+7] == 0 and data[i,t*n_features+9] == 1:
                    data[i,t*n_features+8] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+8] = np.random.binomial(1, 0.5, 1)

    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
    
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    df.index.name = 'subject_id'

    for t in range(n_time_points):
        df.drop(columns=["X0__"+str(t)], inplace=True)
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('binomial_2_'+str(n_time_points)+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('binomial_2_'+str(n_time_points)+'_target.csv',quoting=1)

def generate_binomial_3(n_instances,n_time_points):
    n_features=5

    data = np.zeros([n_instances, n_features*n_time_points])
    data[:,0] = np.random.binomial(1, 0.5, n_instances)
    data[:,1] = np.random.binomial(1, 0.5, n_instances)
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            if data[i,0] == 0:
                data[i,2] = np.random.binomial(1, 0.9, 1)
                data[i,3] = np.random.binomial(1, 0.7, 1)
            else:
                data[i,2] = np.random.binomial(1, 0.1, 1)
                data[i,3] = np.random.binomial(1, 0.3, 1)
                
            if data[i,1] == 0:
                data[i,4] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,4] = np.random.binomial(1, 0.1, 1)
                
                
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.1, 1)
                
                if data[i,t*n_features+5] == 0:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+8] = np.random.binomial(1, 0.7, 1)
                else:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+8] = np.random.binomial(1, 0.3, 1)
                    
                if data[i,t*n_features+6] == 0:
                    data[i,t*n_features+9] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+9] = np.random.binomial(1, 0.1, 1)
        #LABEL 1
        elif labels[i] == 1:
            if data[i,0] == 0:
                data[i,2] = np.random.binomial(1, 0.1, 1)
                data[i,4] = np.random.binomial(1, 0.7, 1)
            else:
                data[i,2] = np.random.binomial(1, 0.9, 1)
                data[i,4] = np.random.binomial(1, 0.3, 1)
                
            if data[i,1] == 0:
                data[i,3] = np.random.binomial(1, 0.1, 1)
            else:
                data[i,3] = np.random.binomial(1, 0.9, 1)
                
                
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 0:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+5] = np.random.binomial(1, 0.1, 1)
                    
                if data[i,t*n_features+1] == 0:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.6, 1)
                else:
                    data[i,t*n_features+6] = np.random.binomial(1, 0.4, 1)
                
                if data[i,t*n_features+5] == 0:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+9] = np.random.binomial(1, 0.7, 1)
                else:
                    data[i,t*n_features+7] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+9] = np.random.binomial(1, 0.3, 1)
                    
                if data[i,t*n_features+6] == 0:
                    data[i,t*n_features+8] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+8] = np.random.binomial(1, 0.9, 1)

    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
    
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    df.index.name = 'subject_id'

    for t in range(n_time_points):
        df.drop(columns=["X0__"+str(t)], inplace=True)
        df.drop(columns=["X1__"+str(t)], inplace=True)
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('binomial_3_'+str(n_time_points)+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('binomial_3_'+str(n_time_points)+'_target.csv',quoting=1)

def generate_multinomial_1(n_instances,n_time_points):
    n_features=3

    values=np.arange(3)
    data = np.zeros([n_instances, n_features*n_time_points])
    uniform=np.ones(len(values))/len(values)
    data[:,0] = np.random.choice(values,p=uniform, size=n_instances)
        
    
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            if data[i,0] == 2:
                data[i,1] = np.random.choice(values,p=[0.9,0.05,0.05])
            elif data[i,0] == 0:
                data[i,1] = np.random.choice(values,p=[0.05,0.05,0.9])
            else:
                data[i,1] = np.random.choice(values,p=[0.05,0.9,0.05])
            
            if data[i,0] == 2:
                data[i,2] = np.random.choice(values,p=uniform)
            elif data[i,0] == 0:
                data[i,2] = np.random.choice(values,p=uniform)
            else:
                data[i,2] = np.random.choice(values,p=uniform)
            
            
            #THIS FOR TIME SLICE 
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 2 and data[i,t*n_features+1] == 0:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.9,0.05,0.05])
                    data[i,t*n_features+4] = np.random.choice(values,p=[0.05,0.05,0.9])
                elif data[i,t*n_features] == 0 and data[i,t*n_features+1] == 2:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.05,0.9,0.05])
                    data[i,t*n_features+4] = np.random.choice(values,p=[0.05,0.9,0.05])
                elif data[i,t*n_features] == 1 and data[i,t*n_features+1] == 1:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.05,0.05,0.9])
                    data[i,t*n_features+4] = np.random.choice(values,p=[0.9,0.05,0.05])
                else:
                    data[i,t*n_features+3] = np.random.choice(values,p=uniform)
                    data[i,t*n_features+4] = np.random.choice(values,p=uniform)
                    
                if data[i,t*n_features+3] == 2:
                    data[i,t*n_features+5] = np.random.choice(values,p=uniform)
                elif data[i,t*n_features+3] == 0:
                    data[i,t*n_features+5] = np.random.choice(values,p=uniform)
                else:
                    data[i,t*n_features+5] = np.random.choice(values,p=uniform)
            
        #LABEL 1
        elif labels[i] == 1:
            if data[i,0] == 2:
                data[i,2] = np.random.choice(values,p=[0.9,0.05,0.05])
            elif data[i,0] == 0:
                data[i,2] = np.random.choice(values,p=[0.05,0.05,0.9])
            else:
                data[i,2] = np.random.choice(values,p=[0.05,0.9,0.05])
            
            if data[i,0] == 2:
                data[i,1] = np.random.choice(values,p=uniform)
            elif data[i,0] == 0:
                data[i,1] = np.random.choice(values,p=uniform)
            else:
                data[i,1] = np.random.choice(values,p=uniform)
            
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 2 and data[i,t*n_features+2] == 0:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.9,0.05,0.05])
                    data[i,t*n_features+5] = np.random.choice(values,p=[0.05,0.05,0.9])
                elif data[i,t*n_features+0] == 0 and data[i,t*n_features+2] == 2:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.05,0.9,0.05])
                    data[i,t*n_features+5] = np.random.choice(values,p=[0.05,0.9,0.05])
                elif data[i,t*n_features] == 1 and data[i,t*n_features+2] == 1:
                    data[i,t*n_features+3] = np.random.choice(values,p=[0.05,0.05,0.9])
                    data[i,t*n_features+5] = np.random.choice(values,p=[0.9,0.05,0.05])
                else:
                    data[i,t*n_features+3] = np.random.choice(values,p=uniform)
                    data[i,t*n_features+5] = np.random.choice(values,p=uniform)
                    
                if data[i,t*n_features+3] == 2:
                    data[i,t*n_features+4] = np.random.choice(values,p=uniform)
                elif data[i,t*n_features+4] == 0:
                    data[i,t*n_features+4] = np.random.choice(values,p=uniform)
                else:
                    data[i,t*n_features+4] = np.random.choice(values,p=uniform)

    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
            
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    
    df.index.name = 'subject_id'    
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('multinomial_1_'+str(n_time_points)+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('multinomial_1_'+str(n_time_points)+'_target.csv',quoting=1)

def generate_multinomial_2(n_instances,n_time_points):
    n_features=4
    
    values=np.arange(3)
    data = np.zeros([n_instances, n_features*n_time_points])
    uniform=np.ones(len(values))/len(values)
        
    
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            data[i,0] = np.random.choice(values,p=uniform)
            if data[i,0] == 2:
                data[i,2] = np.random.choice(values,p=[0.9,0.05,0.05])
            elif data[i,0] == 0:
                data[i,2] = np.random.choice(values,p=[0.05,0.05,0.9])
            else:
                data[i,2] = np.random.choice(values,p=uniform)
            
            data[i,1] = np.random.choice(values,p=uniform)
            data[i,3] = np.random.choice(values,p=uniform)
            
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                if data[i,t*n_features] == 2 and data[i,t*n_features+2] == 0:
                    data[i,t*n_features+4] = np.random.choice(values,p=[0.9,0.05,0.05])
                    data[i,t*n_features+6] = np.random.choice(values,p=[0.05,0.05,0.9])
                elif data[i,t*n_features] == 0 and data[i,t*n_features+2] == 2:
                    data[i,t*n_features+4] = np.random.choice(values,p=[0.05,0.05,0.9])
                    data[i,t*n_features+6] = np.random.choice(values,p=[0.9,0.05,0.05])
                else:
                    data[i,t*n_features+4] = np.random.choice(values,p=uniform)
                    data[i,t*n_features+6] = np.random.choice(values,p=uniform)
                    
                data[i,t*n_features+5] = np.random.choice(values,p=uniform)
                data[i,t*n_features+7] = np.random.choice(values,p=uniform)
            
        #LABEL 1
        elif labels[i] == 1:
            data[i,1] = np.random.choice(values,p=uniform)
            if data[i,1] == 2:
                data[i,3] = np.random.choice(values,p=[0.9,0.05,0.05])
            elif data[i,1] == 0:
                data[i,3] = np.random.choice(values,p=[0.05,0.05,0.9])
            else:
                data[i,3] = np.random.choice(values,p=uniform)
            
            data[i,0] = np.random.choice(values,p=uniform)
            data[i,2] = np.random.choice(values,p=uniform)
            
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                if data[i,t*n_features+1] == 2 and data[i,t*n_features+3] == 0:
                    data[i,t*n_features+5] = np.random.choice(values,p=[0.9,0.05,0.05])
                    data[i,t*n_features+7] = np.random.choice(values,p=[0.05,0.05,0.9])
                elif data[i,t*n_features+1] == 0 and data[i,t*n_features+3] == 2:
                    data[i,t*n_features+5] = np.random.choice(values,p=[0.05,0.05,0.9])
                    data[i,t*n_features+7] = np.random.choice(values,p=[0.9,0.05,0.05])
                else:
                    data[i,t*n_features+5] = np.random.choice(values,p=uniform)
                    data[i,t*n_features+7] = np.random.choice(values,p=uniform)
                    
                data[i,t*n_features+4] = np.random.choice(values,p=uniform)
                data[i,t*n_features+6] = np.random.choice(values,p=uniform)
                
    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
            
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    
    df.index.name = 'subject_id'    
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('multinomial_2_'+str(n_time_points)+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('multinomial_2_'+str(n_time_points)+'_target.csv',quoting=1)

def generate_multiclass(n_instances,n_time_points):
    n_features=10
    n_values = 4
    values=np.arange(n_values)
    classes=np.arange(6)
    data = np.zeros([n_instances, n_features*n_time_points])
    uniform=np.ones(n_values)/n_values
    uniform_class=np.ones(len(classes))/len(classes)
        
    for i in range(n_instances):
        for j in range(n_features*n_time_points):
            data[i,j] = np.random.choice(values,p=uniform)
    
    
    labels = np.zeros([n_instances, 1])
    
    for i in range(0,n_instances):
        labels[i] = np.random.choice(classes,p=uniform_class)
        
        #LABEL 0
        if labels[i] == 0:
            data[i,0] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,2] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            data[i,3] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+0] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+2] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
                data[i,t*n_features+n_features+3] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            
        #LABEL 1
        elif labels[i] == 1:
            data[i,0] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+0] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                
        #LABEL 2
        elif labels[i] == 2:
            data[i,2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,4] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,5] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+4] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+5] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
        #LABEL 3
        elif labels[i] == 3:
            data[i,2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+2] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+3] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
        #LABEL 4
        elif labels[i] == 4:
            data[i,4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,6] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            data[i,7] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+6] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
                data[i,t*n_features+n_features+7] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
                
        #LABEL 5
        elif labels[i] == 5:
            data[i,4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+4] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+5] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                
        #LABEL 6
        elif labels[i] == 6:
            data[i,6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,8] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            data[i,9] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+8] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
                data[i,t*n_features+n_features+9] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
                
        #LABEL 7
        elif labels[i] == 7:
            data[i,7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,9] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+6] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+7] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+9] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                
        #LABEL 8
        elif labels[i] == 8:
            data[i,0] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,1] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,9] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+0] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+1] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+9] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                
        #LABEL 9
        elif labels[i] == 9:
            data[i,0] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
            data[i,1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
            data[i,8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
            data[i,9] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])
            
            #THIS FOR TIME SLICE 1
            for t in range(n_time_points-1):
                data[i,t*n_features+n_features+0] = np.random.choice(values,p=[0.05,0.05,0.05,0.85])
                data[i,t*n_features+n_features+1] = np.random.choice(values,p=[0.85,0.05,0.05,0.05])
                data[i,t*n_features+n_features+8] = np.random.choice(values,p=[0.05,0.05,0.85,0.05])
                data[i,t*n_features+n_features+9] = np.random.choice(values,p=[0.05,0.85,0.05,0.05])

                
    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
            
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    
    df.index.name = 'subject_id'    
       
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('multiclass_'+str(len(classes))+'_parsed.csv',quoting=1)
            
    labels_df.to_csv('multiclass_'+str(len(classes))+'_target.csv',quoting=1)


def generate_binomial_4(n_instances,n_time_points):
    n_features=10

    data = np.zeros([n_instances, n_features*n_time_points])
    labels = np.zeros([n_instances, 1])
    
    for j in range(n_features*n_time_points):
        data[:,j] = np.random.binomial(1, 0.5, n_instances)
    
    for i in range(0,n_instances):
        labels[i] = np.random.binomial(1, 0.5, 1)
        
        #LABEL 0
        if labels[i] == 0:
            if data[i,0] == 0:
                data[i,1] = np.random.binomial(1, 0.1, 1)
            else:
                data[i,1] = np.random.binomial(1, 0.9, 1)
                
            if data[i,2] == 0:
                data[i,3] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,3] = np.random.binomial(1, 0.1, 1)
                
                
            for t in range(n_time_points-1):
                if data[i,t*n_features+0] == 0 and data[i,t*n_features+1] == 0:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.9, 1)
                elif data[i,t*n_features+0] == 1 and data[i,t*n_features+1] == 1:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.1, 1)
                else:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.5, 1)
                    
                if data[i,t*n_features+2] == 0 and data[i,t*n_features+3] == 1:
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features+2] == 1 and data[i,t*n_features+3] == 0:
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.5, 1)
        #LABEL 1
        elif labels[i] == 1:
            if data[i,0] == 0:
                data[i,3] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,3] = np.random.binomial(1, 0.1, 1)
                
            if data[i,1] == 0:
                data[i,2] = np.random.binomial(1, 0.9, 1)
            else:
                data[i,2] = np.random.binomial(1, 0.1, 1)
                
                
            for t in range(n_time_points-1):                
                if data[i,t*n_features+0] == 0 and data[i,t*n_features+3] == 1:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features+0] == 1 and data[i,t*n_features+3] == 0:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+n_features+0] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+n_features+3] = np.random.binomial(1, 0.5, 1)
                    
                if data[i,t*n_features+1] == 0 and data[i,t*n_features+2] == 1:
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.9, 1)
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.1, 1)
                elif data[i,t*n_features+1] == 1 and data[i,t*n_features+2] == 0:
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.1, 1)
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.9, 1)
                else:
                    data[i,t*n_features+n_features+1] = np.random.binomial(1, 0.5, 1)
                    data[i,t*n_features+n_features+2] = np.random.binomial(1, 0.5, 1)
    col = []
    for t in range(n_time_points):
        for f in range(n_features):
            col.append("X"+str(f)+"__"+str(t))
    
    df = pd.DataFrame(data=data,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=col)
    df.index.name = 'subject_id'
   
    labels_df = pd.DataFrame(data=labels,    # values
                          index=list(range(n_instances)),    # 1st column as index
                          columns=['label'])
    labels_df.index.name = 'subject_id'
    
    df.to_csv('binomial_joao_parsed.csv',quoting=1)
            
    labels_df.to_csv('binomial_joao_target.csv',quoting=1)