#parse command line inputs expecting:
#python train.py sw/lw                      #for random anns
#python train.py sw/lw n_params n_layers    #for benchmark anns
import sys
args = sys.argv
assert (len(args) == 2 or len(args) == 4), 'Incorrect input format'
bench = len(args) == 4
wvl_region = args[1]
if bench:
    params = int(args[2])
    layers = int(args[3])
if wvl_region == 'sw':
    n_outputs = 3
else:
    n_outputs = 1

#set up tensorflow/keras:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from neural_networks import RandomANN, benchmark_mlp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from uuid import uuid4

#build the network:
print('Generating ANN',flush=True)
if not bench:
    ann = RandomANN(nio = [9,n_outputs])
    ann.build()
    ann = ann.ann
    fname = './data/anns/' + wvl_region + '/random/' + uuid4().hex
else:
    ann,_ = benchmark_mlp(layers,params,9,n_outputs)
    fname = './data/anns/' + wvl_region + '/benchmark/L' + str(layers) + '_P' + str(params) + '_' + uuid4().hex
    
ann.summary()
ann.compile(optimizer=Adam(learning_rate = 0.001),loss = 'MSE', metrics = ['MAE'])
def lr_schedule(epoch,lr):
    return 10**-(3+epoch//3)

print('Loading Data',flush=True)
import numpy as np
inputs = np.load('./data/training/' + wvl_region + '_inputs.npy')
targets = np.load('./data/training/' + wvl_region + '_targets.npy')

print('Training',flush=True)
ann.fit(inputs,targets,batch_size = 64,verbose=2,epochs=10,
                  callbacks=[LearningRateScheduler(lr_schedule)])
ann.save(fname)
