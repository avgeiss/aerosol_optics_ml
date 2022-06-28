#!/usr/bin/env python3
#
#Andrew Geiss, Feb 15th 2022
#
#Code for building neural networks in keras. The RandomAnn class contains
#functions that can create randomly wired neural networks using keras and has some
#additional convenient functions: saving and loading the random networks without
#building a tensorflow graph, a training subroutine, a function to count the
#trainable parameters in the random network. The benchmark_mlp function builds
#a simple feed-forward fully connected network to use for comparison to the random
#networks. it allows for requesting a layer count and approximate total
#parameter count and handles the rest itself.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from uuid import uuid4
from tensorflow.keras.layers import Dense, Input, add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class RandomANN:
    
    #this class defines randomly wired fully connected neural network archiectures
      
    layer_count_range = (2,12)
    layer_size_range = (7,45)
        
    def __init__(self, nio = [9,3], load_fname=None):
        
        if load_fname is None:
            #generate the metadata that defines a new random ANN
            self.n_layers = np.random.randint(*self.layer_count_range)                              #choose the number of layers
            self.n_neurons = int(np.random.randint(*self.layer_size_range)*(12/self.n_layers))      #choose layer sizes
            self.adjacency = self.__random_adj_mat__()                                              #build a random adjacency matrix
            self.name = uuid4().hex
            self.merge_types = [np.random.choice(['add','cat']) for _ in range(self.n_layers)]
            self.loss = []
            
        else:
            #load the metadata from a file (does not load any weights!)
            ann_data = np.load(load_fname)
            fields = ['n_neurons','adjacency','loss','merge_types']
            self.n_neurons, self.adjacency, self.loss, self.merge_types = [ann_data[f] for f in fields]
            self.n_layers = self.adjacency.shape[0]
            self.name = load_fname.split('/')[-1].split('.')[0]
            self.loss = list(self.loss)
        self.n_in, self.n_out = nio[0], nio[1]
        
    def __random_adj_mat__(self):
        #generate a random acyclic adjacency matrix:
        adj = np.zeros((self.n_layers,self.n_layers))
        x,y = np.tril_indices(self.n_layers)
        edges = np.zeros(x.shape)
        n_active = np.random.randint(0,x.shape[0]+1)
        edges[np.random.choice(np.arange(len(edges)), n_active, replace=False)] = 1
        adj[x,y] = edges
        
        #make sure each node has at least one inbound and one outbound edge:
        for i in range(self.n_layers):
            if np.max(adj[i,:]) == 0:
                adj[i,np.random.randint(0,i+1)] = 1
            if np.max(adj[:,i]) == 0:
                adj[np.random.randint(i,self.n_layers),i] = 1
                
        return adj
    
    def count_params(self):
        #returns a count of the ANN's weights and biases based on the metadata
        #(without needing to build a tf graph)
        count = (self.n_in+1)*self.n_neurons
        for i in range(self.n_layers):
            if self.merge_types[i] == 'add':
                insz = self.n_neurons
            else:
                insz = self.n_neurons*np.sum(self.adjacency[i,:])
            if i == self.n_layers-1:
                outsz = self.n_out
            else:
                outsz = self.n_neurons
            count += insz*outsz + outsz
        return int(count)
    
    def build(self):
        #build the ann using keras:
        xin = Input((self.n_in,))
        if self.n_neurons>self.n_in:
            chan_pad = Dense(self.n_neurons-self.n_in,activation='tanh')(xin)
            layers = [concatenate([xin,chan_pad])]
        else:
            layers = [Dense(self.n_neurons)(xin)]
        
        #iterate over each layer
        for i in range(0,self.n_layers):
            
            #gather inputs:
            inputs = [layers[ind] for ind in list(np.where(self.adjacency[i,:])[0])]
            if len(inputs)>1:
                if self.merge_types[i] == 'add':
                    inputs = add(inputs)
                else:
                    inputs = concatenate(inputs)
            else:
                inputs = inputs[0]
                
            #create layer:
            if i < self.n_layers-1:
                x = Dense(self.n_neurons, activation='tanh')(inputs)
                layers.append(x)
            else:
                xout = Dense(self.n_out,activation='sigmoid')(inputs)
        self.ann = Model(xin,xout)
        self.ann.compile(optimizer=Adam(learning_rate=0.001),loss='MSE')
        
    def save(self,save_dir):
        np.savez(save_dir + self.name, merge_types=self.merge_types, 
                 n_neurons=self.n_neurons, adjacency=self.adjacency,
                 loss=self.loss)
        
    def disp(self):
        print('Randomly Constructed ANN Summary:')
        print('Weight + Bias Count: ' + str(self.count_params()))
        print('Connectivity: ' + str(int(np.sum(self.adjacency))) + '/' + str(int((self.n_layers**2 + self.n_layers)/2)))
        print('Layer Sizes: ' + str(self.n_neurons))
    
    def plot(self,output_dir='./'):
        plot_model(self.ann,output_dir + self.name + '.png',show_shapes=True)
    
    def train(self,x,y,batch_size=32):
        hist = self.ann.fit(x,y,batch_size=batch_size,epochs=1,verbose=0)
        self.loss.append(hist.history['loss'][0])
        
        
#This function generates a basic feed forward multi-layer perceptron
#it allows specification of a numer of layers and an approximate requested total
#parameter count and automatically scales the number of neurons per layer.
def benchmark_mlp(layers, params, inputs, outputs):
    
    def param_count(layers,neurons,inputs,outputs):
        count = inputs*neurons + neurons
        count += (neurons**2 + neurons)*(layers-1)
        count += neurons*outputs+outputs
        return count
    
    def determine_layer_sizes(layers,params,inputs,outputs):
        neurons = 0
        cur_params = 0
        while cur_params < params:
            neurons += 1
            cur_params = param_count(layers, neurons, inputs, outputs)
        return neurons
    
    neurons = determine_layer_sizes(layers, params, inputs, outputs)
    
    xin = Input((inputs,))
    x = xin
    for i in range(layers):
        x = Dense(neurons,activation='tanh')(x)
    x = Dense(outputs,activation='sigmoid')(x)
    ann = Model(xin,x)
    ann.compile(optimizer=Adam(learning_rate=0.001),loss='MSE')
    return ann, param_count(layers, neurons, inputs,outputs)
