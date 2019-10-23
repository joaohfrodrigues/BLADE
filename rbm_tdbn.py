# -*- coding: utf-8 -*-
"""
Multivariate time-series analysis model using Restricted Boltzmann machines 
(RBM) and temporal dynamic Bayesian networks (tDBN)
Created on Tue Jun 11 2019

@author: João Henrique Rodrigues, IST

version: 1.0
"""
import sys, os
import numpy as np
import torch
from rbm_categorical import CategoricalRBM as cRBM
import pandas as pd
from rbm_dataset import RBMDataset
import rbm_utils
#weight analysis
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

#MACROS
PREDICTION = 'p'
LEARNING = 'l'

TEST = 1
VAL = 0

CV_K = 5

class RBMtDBN ():
    ''' Class implementing the required methods for running an RBM and tDBN model.
    '''
    def __init__(self, parsed_args):
        '''Initialization method for the model.
        
        Parameters
        ----------
        model_type : char
            Defines the application of the model: learning, predction.
        hidden_units: int
            Number of hidden units to be used in the Boltzmann machines.
        batch_size_ratio : float, default = 0.2
            Ratio of the dataset to be used as batch.
        cd : int, default = 1
            Defines the k in the CD-k algorithm.
        epochs: int
            Define the number of epochs used in the learning of the Boltzmann 
            Machines.
        filepath: string
            Path of the file which will also be used to find the labels of the 
            data.
        learning_rate: float
            Learning rate of the Boltzmann machines.
        weight_decay: float
            Weights decay of the Boltzmann machines.
        test_set_ratio: float
            Ratio of the dataset to be set apart for testing.
        validation_set_ratio: float
            Ratio of the dataset to be set apart for validation of the model's 
            parameters.
        persistent: bool
            Boolean stating whether or not to use persistent CD.
        verbose: bool
            Additional printing.
        '''
        self.model_type = parsed_args.model_type
        self.hidden_units = parsed_args.hidden_units
        self.batch_size_ratio = parsed_args.batch_size_ratio
        self.cd = parsed_args.contrastive_divergence #0 - PCD, k - CD-k
        self.epochs = parsed_args.epochs
        self.filepath = parsed_args.data_file
        self.learning_rate = parsed_args.learning_rate
        self.weight_decay = parsed_args.weight_decay
        self.test_set_ratio= parsed_args.test_set_ratio
        self.validation_set_ratio= parsed_args.validation_set_ratio
        self.persistent = parsed_args.persistent_cd
        self.verbose = parsed_args.verbose
    
    def load_dataset(self):
        '''Loads the dataset and creates the data structures needed for 
        learning, according to the model type that is indicated by the user.
        
        Working modes
        ----------
        Learning:
            Use all available data to train an RBM, then use the RBM to
            infer the hidden units from the visible units. Train a tDBN with
            the inferred values.
        Prediction:
            Split the data into training and test sets. The training set is
            also splitted by label, training one RBM per label. The traned RBMs
            are used to infer the hidden units from the test set, for each
            label. The training set will also be used to train a tDBN, on which
            the test set with the hidden units is applied. The tDBN algorithm
            outputs a set of probabilities that represent the probability of
            each subject being assigned a specific label. These results are
            used to perform classification.
        '''
        if self.verbose : print('Loading dataset...')
        
        #read original data
        self.train_loaders = []
        df = pd.read_csv(self.filepath + '_parsed.csv', index_col= 'subject_id', header=0)
        feat,time = df.columns[-1].split('__')
        feat_0,time_0 = df.columns[0].split('__')
        feat = feat[1:]
        feat_0 = feat_0[1:]
        self.features = int(feat) + 1 - int(feat_0)
        self.time_points = int(time) + 1 - int(time_0)
        self.instances = len(df)
        self.visible_units = self.features
        self.ext_instances = self.time_points*self.instances #number of intances after one-hot encoding
        
        if model_type == PREDICTION:
            self.labels, self.label_values, self.label_indices = rbm_utils.count_labels(self.filepath)
            self.n_labels = len(self.label_values)
            input_labels = np.array(self.labels['label'])
        else: #learning doesn't require labeled data
            self.labels = np.zeros(self.instances*self.time_points)
            input_labels = self.labels
            self.n_labels = 1
        
        #create an initial dataset
        self.dataset = RBMDataset(df, self.features, self.time_points, labels=np.array(input_labels))
        self.sum_visible = sum(self.dataset.n_diff)
        self.batch_size = int(np.floor(self.ext_instances*self.batch_size_ratio))
        
        if self.model_type == LEARNING:
            #learning requires just a training set, since its results can't be compared to anything
            self.train_loaders.append([])
            self.train_loaders[0] = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        elif self.model_type == PREDICTION:
            #prediction requires a training set for each label and a general test set
            #mind that here the initial datafram is being used for splitting and not the dataset
            _, _, _, self.pre_train_indices, _, self.test_indices = rbm_utils.create_train_sets(df,self.label_indices, test_train_ratio=self.test_set_ratio, validation_ratio=0,batch_size=self.batch_size, get_indices=True, label_ratio = True)
        
        if model_type == PREDICTION:
            by_label_test_df = []
            self.by_label_test_dataset = []
            self.by_label_test_instances = []
            self.by_label_test_indices = []
            self.by_label_test_batch_size = []
            self.train_test_loaders = []
            self.by_label_test_batch_size = []
                
            self.pre_train_df = pd.DataFrame(data=df.values[self.pre_train_indices-df.index[0]],    # values
                                               index=list(range(len(self.pre_train_indices))),    # 1st column as index
                                               columns=df.columns)
            self.pre_train_labels = np.array(self.labels['label'][self.pre_train_indices-df.index[0]])
            
            
            #THIS IS FOR TESTING AND NOT VALIDATING
            for label in range(self.n_labels):
                by_label_test_df.append([])
                self.by_label_test_dataset.append([])
                self.by_label_test_instances.append([])
                self.by_label_test_indices.append([])
                self.train_test_loaders.append([])
                self.by_label_test_batch_size.append([])
                
                #get the indicies corresponding to each label
                self.by_label_test_indices[label] = [item for item in self.label_indices[label] if item in self.pre_train_indices]
                #create a dataframe and dataset
                by_label_test_df[label] = pd.DataFrame(data=df.values[self.by_label_test_indices[label]-df.index[0]],    # values
                                                 index=list(range(len(self.by_label_test_indices[label]))),    # 1st column as index
                                                 columns=df.columns)
                self.by_label_test_instances[label] = len(by_label_test_df[label])
                self.by_label_test_dataset[label] = RBMDataset(by_label_test_df[label], self.features, self.time_points, labels=np.array(self.labels['label'][self.by_label_test_indices[label]-df.index[0]]), n_diff = self.dataset.n_diff)
            


                #creating the data loaders for the training set
                self.by_label_test_batch_size[label] = int(np.floor(self.by_label_test_instances[label]*self.time_points*self.batch_size_ratio))                
                
                #resctrict size of batch for large sets
                if self.by_label_test_batch_size[label] > 100:
                    self.by_label_test_batch_size[label] = 100
                
                self.train_test_loaders[label] = torch.utils.data.DataLoader(self.by_label_test_dataset[label], batch_size=self.by_label_test_batch_size[label])
            
            #create a dataframe and dataset for the test set
            test_df = pd.DataFrame(data=df.values[self.test_indices-df.index[0]],    # values
                                               index=list(range(len(self.test_indices))),    # 1st column as index
                                               columns=df.columns)
            self.test_labels = np.array(self.labels['label'][self.test_indices-df.index[0]])
            self.test_dataset = RBMDataset(test_df, self.features, self.time_points, labels=self.test_labels, n_diff = self.dataset.n_diff)
            
            #creating the data loader for the test set
            self.test_batch_size = int(np.floor(len(test_df)*self.time_points*self.batch_size_ratio))
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.test_batch_size)
        return df
        
    def create_rbms(self, current=-1, total=-1):  
        self.rbms = []
        self.sum_data = []
        self.train_val_indices = []
        self.train_indices = []
        self.train_labels = np.zeros(len(self.pre_train_indices))
        
        for label in range(self.n_labels):
            self.train_val_indices.append([])
            for i in range(len(self.pre_train_indices)):
                if self.pre_train_indices[i] in self.label_indices[label]:
                    self.train_val_indices[label].append(i)
                    self.train_labels[i] = self.labels.values[self.pre_train_indices[i]-df.index[0]]
        
        #self.train_split = int(np.floor(self.pre_train_df.shape[0])*(total-1)/total)
        self.val_split = int(np.floor(self.pre_train_df.shape[0])/total)
        
        if total >= 0:
            self.train_indices = self.train_val_indices[0:self.val_split*current] + self.train_val_indices[self.val_split*(current+1):len(self.train_val_indices)]
            self.val_indices = self.train_val_indices[self.val_split*current:self.val_split*(current+1)]
        else:
            _, _, _, self.train_indices, _, self.val_indices = rbm_utils.create_train_sets(self.pre_train_df,self.train_val_indices, test_train_ratio=self.validation_set_ratio, validation_ratio=0,batch_size=self.batch_size, get_indices=True, label_ratio = True)
        
        if model_type == PREDICTION:
            if self.verbose : print('Creating by_label_datasets...')
            #for prediction, it's necessary to create 1 training set per label
            by_label_df = []
            self.by_label_dataset = []
            self.by_label_instances = []
            self.by_label_indices = []
            self.by_label_batch_size = []
            
            for label in range(self.n_labels):
                by_label_df.append([])
                self.by_label_dataset.append([])
                self.by_label_instances.append([])
                self.by_label_indices.append([])
                
                #get the indicies corresponding to each label
                self.by_label_indices[label] = [item for item in self.train_val_indices[label] if item in self.train_indices]
                #create a dataframe and dataset
                by_label_df[label] = pd.DataFrame(data=self.pre_train_df.values[self.by_label_indices[label]],    # values
                                                 index=list(range(len(self.by_label_indices[label]))),    # 1st column as index
                                                 columns=df.columns)
                self.by_label_instances[label] = len(by_label_df[label])
                self.by_label_dataset[label] = RBMDataset(by_label_df[label], self.features, self.time_points, labels=np.array(self.train_labels[self.by_label_indices[label]]), n_diff = self.dataset.n_diff)
                
                
            #create a dataframe and dataset for the validation set
            val_df = pd.DataFrame(data= self.pre_train_df.values[self.val_indices],    # values
                                  index=list(range(len(self.val_indices))),    # 1st column as index
                                  columns=df.columns)
            

            self.val_labels = np.array(self.pre_train_labels[self.val_indices])
            self.val_dataset = RBMDataset(val_df, self.features, self.time_points, labels=self.val_labels, n_diff = self.dataset.n_diff)
        
        ###### RBM CREATION ######
        if self.verbose : print('Creating RBMs...')
        if model_type == LEARNING:
            #only one RBM is needed for learning
            self.rbms.append([])
            sum_data = sum(self.dataset.data)/self.dataset.data.shape[0]
            self.rbms[0] = cRBM(self.visible_units, self.hidden_units, self.dataset.n_diff, sum_data, cd=self.cd, persistent=self.persistent, learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        elif model_type == PREDICTION:
            #for prediction, there is one RBM for each label
            for label in range(self.n_labels):
                self.rbms.append([])
                self.train_loaders.append([])
                self.by_label_batch_size.append([])

                sum_data = sum(self.by_label_dataset[label].data)/self.by_label_dataset[label].data.shape[0]
                self.rbms[label] = cRBM(self.visible_units, self.hidden_units, self.dataset.n_diff, sum_data, cd=self.cd, persistent=self.persistent, learning_rate=self.learning_rate, weight_decay=self.weight_decay)

                #creating the data loaders for the training set
                self.by_label_batch_size[label] = int(np.floor(self.by_label_instances[label]*self.time_points*self.batch_size_ratio))                
                
                #resctrict size of batch for large sets
                if self.by_label_batch_size[label] > 100:
                    self.by_label_batch_size[label] = 100
                
                self.train_loaders[label] = torch.utils.data.DataLoader(self.by_label_dataset[label], batch_size=self.by_label_batch_size[label])
            
            self.val_batch_size = int(np.floor(len(val_df)*self.time_points*self.batch_size_ratio))
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size)
        
    
    def train_fixed_rbm(self):
        '''Trains the RBMs of the model, using the CD-k algorithm or
        PCD.
        '''        
        if self.verbose : print('Training RBM...')
        for i in range(len(self.rbms)):
            #last_error = 0.0
            for epoch in range(self.epochs):
                epoch_error = 0.0
            
                for batch, _ in self.train_loaders[i]:
                    batch_error = self.rbms[i].contrastive_divergence(batch)
                    epoch_error += batch_error
                if self.verbose : print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

            '''
            max_weight = torch.max(abs(self.rbms[i].weights))
            self.rbms[i].weights = self.rbms[i].weights * 10/max_weight
            '''
            if self.verbose : print('Printing weights...\n\n')
            if self.verbose : print(self.rbms[i].weights)
            if self.verbose : print('\n\n')
            
            
    
    def extract_features(self, test_val = None):
        ''' Based on the learned RBM, convert the visible features into hidden
        values.
        '''   
        if self.verbose : print('Extracting features...')
        extracted_features = []
        self.extracted_test_features = []
        
        for k in range(self.n_labels):
            extracted_features.append([])
            self.extracted_test_features.append([])
            
            if self.model_type == LEARNING:
                extracted_features[k] = np.zeros((self.instances*self.time_points, self.hidden_units))
        
                for i, (batch, _) in enumerate(self.train_loaders[k]):
                    extracted_features[k][i*self.batch_size:i*self.batch_size+len(batch)] = self.rbms[k].sample_hidden_state(batch).cpu().numpy()
            
            elif self.model_type == PREDICTION:
                if test_val == TEST:
                    test_loader = self.test_loader
                    test_batch_size = self.test_batch_size
                    test_indices = self.test_indices
                    train_instances = self.by_label_test_instances
                    train_loaders = self.train_test_loaders
                    train_batch_size = self.by_label_test_batch_size
                elif test_val == VAL:
                    test_loader = self.val_loader
                    test_batch_size = self.val_batch_size
                    test_indices = self.val_indices
                    train_loaders = self.train_loaders
                    train_instances = self.by_label_instances
                    train_batch_size = self.by_label_batch_size
                    
                extracted_features[k] = np.zeros((train_instances[k]*self.time_points, self.hidden_units))
                self.extracted_test_features[k] = np.zeros((len(test_indices)*self.time_points, self.hidden_units))
                
                for i, (batch, _) in enumerate(train_loaders[k]):
                    extracted_features[k][i*train_batch_size[k]:i*train_batch_size[k]+len(batch)] = self.rbms[k].sample_hidden_state(batch).cpu().numpy()
            
                for i, (batch, _) in enumerate(test_loader):
                    self.extracted_test_features[k][i*test_batch_size:i*test_batch_size+len(batch)] = self.rbms[k].sample_hidden_state(batch).cpu().numpy()
        
        self.write_output(extracted_features, extracted_test_features = self.extracted_test_features, test_val = test_val)
    
    def write_output(self, extracted_features, extracted_test_features = None, test_val = None):
        ''' Method that writes the output files to be used on the tDBN, with the inferred hidden units.
        
        Parameters
        ----------
        extracted_features : list
            Set of hidden units inferred from the whole data in case of
            the learning task or the training set in case of prediction.
        extracted_test_features (optional): list
            Set of hidden units inferred from the test set. This is only used
            for prediction.
            
        Output
        ----------
        One or two files are generated, that'll be used for the training and
        probability calculation using tDBN.
        '''
        csv_data_train = []
        csv_data_test = []
        
        if test_val == TEST:
            test_filepath = "bm_test"
            t_indices = self.test_indices
            by_label_instances = self.by_label_test_instances
            by_label_dataset = self.by_label_test_dataset
        elif test_val == VAL:
            test_filepath = "bm_val"
            t_indices = self.val_indices
            by_label_instances = self.by_label_instances
            by_label_dataset = self.by_label_dataset
        
        for k in range(self.n_labels):
            csv_data_train.append([])
            csv_data_test.append([])
            
            if self.model_type == LEARNING:
                csv_data_train[k] = np.zeros((self.instances,self.hidden_units*self.time_points))
                
                aux_index = 0
                for i in range(self.instances):
                    for j in range(self.time_points):
                        csv_data_train[k][aux_index,(j*self.hidden_units):((j+1)*self.hidden_units)] = extracted_features[k][self.instances*j+i]
                    aux_index+=1
            elif self.model_type == PREDICTION:
                csv_data_train[k] = np.zeros((by_label_instances[k],self.hidden_units*self.time_points))
                csv_data_test[k] = np.zeros((len(t_indices),self.hidden_units*self.time_points))

                aux_index = 0
                for i in range(by_label_instances[k]):
                    for j in range(self.time_points):
                        csv_data_train[k][aux_index,(j*self.hidden_units):((j+1)*self.hidden_units)] = extracted_features[k][by_label_instances[k]*j+i]
                    aux_index+=1        
    
                aux_index = 0
                for i in range(len(t_indices)):
                    for j in range(self.time_points):
                        csv_data_test[k][aux_index,(j*self.hidden_units):((j+1)*self.hidden_units)] = extracted_test_features[k][len(t_indices)*j+i]
                    aux_index+=1
        
        out_columns = []
        
        for t in range(self.time_points):
            for i in range(self.hidden_units):
                out_columns.append('H'+str(i)+'__'+str(t))

        for k in range(self.n_labels):
            write_df = pd.DataFrame(data=csv_data_test[k],    # values
                         index=list(range(len(csv_data_test[k]))),    # 1st column as index
                         columns=out_columns)
            write_df.to_csv('results/'+test_filepath+'_'+str(k)+'.csv',quoting=1)
    

            write_df = pd.DataFrame(data=csv_data_train[k],    # values
                         index=list(range(len(csv_data_train[k]))),    # 1st column as index
                         columns=out_columns)
            write_df.to_csv('results/bm_train_'+str(k)+'.csv',quoting=1)
            
            if test_val == TEST:
                by_label_dataset[k].data_frame.to_csv('results/bm_original_train_'+str(k)+'.csv',quoting=1)
                
        if test_val == TEST:
            self.test_dataset.data_frame.to_csv('results/bm_original_test.csv',quoting=1)
    
    def run_tdbn(self, exe,filepath, out_filepath = None,test_filepath = None, alt_verbose = False):
        ''' Runs the tDBN algorithm, for either the learning or prediction task.
        
        Parameters
        ----------
        filepath : string
            Path to the file which will be used for learning the structure and
            parameters of the DBN.
        test_filepath (optional): string
            Path to file that contains the test set used to calculate the 
            output probabilities when using tDBN.
        out_filepath (optional): string
            Path to the output file which will store the output probabilities
            in the prediction task.
        '''
        if self.verbose : print('\nRunning tDBN with hidden units...\n')
        
        if test_filepath != None:
            args = [exe+'.jar', '-i', filepath, '-f', test_filepath, '-w', out_filepath, '-p','1', '-pm']
        else:
            args = [exe+'.jar', '-i', filepath, '-p','1', '-pm']
            
        result = rbm_utils.jarWrapper(*args)
         
        if alt_verbose == True:
            for line in result:
                print(line)
    
    def get_pred_results(self, filepath, test_val):
        ''' Reads the output files from the tDBN for each class, estimates the
        labels by finding the maximum and compares the results with the known
        labels.
        
        Output
        ----------
        Accuracy of the prediction task.
        '''
        
        if test_val == TEST:
            test_instances=len(self.test_indices)
            true_labels = self.test_labels
            train_instances = self.by_label_test_instances
            total_instances = len(self.pre_train_indices)
        elif test_val == VAL:
            test_instances=len(self.val_indices)
            true_labels = self.val_labels
            train_instances = self.by_label_instances
            total_instances = len(self.train_indices)
        #a=17
        probs_df = np.zeros((test_instances, self.n_labels))

        for k in range(self.n_labels):
            probs_df[:,k] = pd.read_csv(filepath+str(k)+'.csv').values.reshape(test_instances)
            probs_df[:,k] = probs_df[:,k] + np.log10(train_instances[k]/total_instances)
        
        self.pred_labels = np.zeros((test_instances,2))
        for i in range(test_instances):
            self.pred_labels[i,0] = self.label_values[probs_df[i].argmax()]
            self.pred_labels[i,1] = probs_df[i].max()
            
        correct = sum(self.pred_labels[:,0] == true_labels)
        total = len(true_labels)
        print('Accuracy: %d/%d (%.2f) \n' % (correct, total, correct/total))
        return correct/total
        
    def erase_files(self, test_val):
        #os.remove("results/bm_original_test.csv")
        for label in range(self.n_labels):
            #os.remove("results/bm_original_train"+str(label)+".csv")
            os.remove("results/bm_train_"+str(label)+".csv")
            os.remove("results/bm_val_"+str(label)+".csv")
            os.remove("results/probs_"+str(label)+".csv")
    
    def weight_analyzer(self, rbm, h_selected=None, filepath = None):
        ''' Analysis the RBM weights, creates a heatmap visual to help 
        understand which features are most important for each hidden unit.
        
        Parameters
        ----------
        rbm: RestrictedBoltzmannMachine
            RBM from which to exctract the weight information.
        h_selected:
            -
        filepath (optional):
            Path of the image file on which the heatmap will be stored.
        '''
        if self.verbose: print('\nPrinting results of the trained RBM...\n')
        #print(rbm.weights)
        #feat_values = rbm_utils.parse_dic(self.filepath)
        #rbm_utils.weight_analyzer(self.rbm.weights,feat_values,0.5)
        
        sns_plot = sns.heatmap(rbm.weights.t(),center=0,cmap=sns.color_palette("coolwarm", 10))
        sns_plot.set(xlabel='features', ylabel='hidden units')
        
        if filepath is None:
            sns_plot.figure.savefig('fig/weights.pdf')
        else:
            sns_plot.figure.savefig(filepath)
        plt.clf()

"""
Main Code
"""
#RUNNING MODE
#   "dev": run on developing mode
#   "run": run on command line
running_mode="dev"

#MODEL
#   "learning": learn an RBM and tDBN based on the data
#   "prediction": learn an RBM and predict some temporal data using tDBN
model_type=PREDICTION
hidden_units = '3'
batch_size_ratio = '0.1'
cd = '1' #k - CD-k
epochs = '100'
#filepath = 'new_data/ArabicDigits'
filepath = 'example_data/binomial_joao_2_6'
#learning_rate = '0.05'
learning_rate = '0.05'
#weight_decay = '1e-2'
weight_decay = '1e-4'
test_set_ratio= '0.2'
validation_set_ratio= '0.2'
persistent = 'False'
verbose = 'False'
n_runs = '10'

input_args=['rbm_tdbn.py', model_type, filepath, 
            '-hu',hidden_units, '-e', epochs, 
            '-bs', batch_size_ratio,
            '-lr', learning_rate, '-wd', weight_decay,
            '-tsr', test_set_ratio, '-vsr', validation_set_ratio,
            '-nr', n_runs]

if __name__ == "__main__":
    parser = rbm_utils.create_parser()
    val_runs = 5
    rbm_usage = True
    if running_mode == "run":
        input_args = sys.argv
        model_type = input_args[1]
    
    parsed_args = parser.parse_args(input_args[1:])
    acc = np.zeros(parsed_args.number_runs)
    original_acc = np.zeros(parsed_args.number_runs)
    val_rbms = [None] * val_runs
    
    for n in range(parsed_args.number_runs):
        network = RBMtDBN(parsed_args)
        df = network.load_dataset()
        val_acc = np.zeros(val_runs)
        
        if model_type == PREDICTION:
            if rbm_usage:
                for j in range(val_runs):
                    print("val run = " + str(j))
                    
                    network.create_rbms(j, -1)
                    
                    network.train_fixed_rbm()
                    #print(network.rbms[0].weights.max())
                    #print(network.rbms[0].weights)
                    val_rbms[j] = network.rbms.copy()
                    for k in range(val_runs):
                        network.extract_features(VAL)
                        for i in range(len(network.rbms)):
                            network.run_tdbn(exe='tDBN_bin',filepath = 'results/bm_train_'+str(i)+'.csv', out_filepath = 'results/probs_'+str(i)+'.csv', test_filepath = 'results/bm_val_'+str(i)+'.csv')
                        val_acc[j] = val_acc[j] + network.get_pred_results('results/probs_', VAL)
                        network.erase_files(VAL)
                print(val_acc)
                network.rbms = val_rbms[val_acc.argmax()]
                network.extract_features(TEST)
                for i in range(len(network.rbms)):
                    network.weight_analyzer(network.rbms[i], filepath = 'fig/weights_'+str(i)+'.pdf')
                    network.run_tdbn(exe='tDBN_bin',filepath = 'results/bm_train_'+str(i)+'.csv', out_filepath = 'results/probs_'+str(i)+'.csv', test_filepath = 'results/bm_test_'+str(i)+'.csv')
                    network.run_tdbn(exe='tDBN_cat',filepath = 'results/bm_original_train_'+str(i)+'.csv', out_filepath = 'results/original_probs_'+str(i)+'.csv', test_filepath = 'results/bm_original_test.csv')
            else:
                network.extract_features(TEST)
                for i in range(len(network.rbms)):
                    network.run_tdbn(exe='tDBN_cat',filepath = 'results/bm_original_train_'+str(i)+'.csv', out_filepath = 'results/original_probs_'+str(i)+'.csv', test_filepath = 'results/bm_original_test.csv')
            
        elif model_type == LEARNING:
            network.train_fixed_rbm()
            network.extract_features()
            network.run_tdbn(filepath = 'results/bm_train_'+str(i)+'.csv')
            

        if model_type == PREDICTION:
            if rbm_usage:
                acc[n] = network.get_pred_results('results/probs_', TEST)
            original_acc[n] = network.get_pred_results('results/original_probs_', TEST)
        
        print('\n' + str(n) + '\n')
    
    if rbm_usage:
        print(acc)
    print(original_acc)