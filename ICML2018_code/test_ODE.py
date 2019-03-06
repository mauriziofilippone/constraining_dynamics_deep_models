## Copyright 2019 Marco Lorenzi and Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import torch
import torch.optim
import numpy as np
from torch.autograd import Variable
import PyTorchDGP
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import misc
import time
import sys
import Optimize

## The variable dataset can be any of the following "Lotka-Volterra", "Fitz-Hugh-Nagumo", "protein_trans_pathways"
## The name of the dataset is taken as argument from the command line
dataset = str(sys.argv[1])

## The variable index_experiment can be either 0 or 1 
## It selects one of two possible configurations that were used to generate the datasets used in the experiments and it is taken as argument from the command line
index_experiment = int(sys.argv[3])

## The variable fold can be any integer between 0 and 4 
## It selects one of five possible folds that were used to validate the competing methods
fold = int(sys.argv[2])

## Establish the root of the filenames pertaining to a given combination of dataset, experiment and fold
filename_root = "FOLDS/" + dataset + '_EXPERIMENT_' + str(index_experiment)

## Read data
x,y,coefs,initial_cond, tmin, tmax = misc.read_data(filename_root, fold)

## Initialize GP parameters sigma (marginal st dev) and l (length-scale)
sigma = float(np.log(torch.max(y).data.numpy()-torch.min(y).data.numpy()))
l = float(np.log(torch.max(x).data.numpy()-torch.min(x).data.numpy()))
print("Initial covariance parameters of GPs: \nlog-length-scale", l, "\nlog-marginal standard deviation", sigma, "\n") 

## Initialize noise on observations
noise = float(np.log((torch.max(y).data.numpy()-torch.min(y).data.numpy())/1e5))
print("Initial log-standard deviation noise on observations:", noise) 

## Set the right nuber of parameters for a given ODE
if dataset == 'Lotka-Volterra':
    N_ODE_parameters = 4
    D_OUT = 2
elif dataset == 'Fitz-Hugh-Nagumo':
    N_ODE_parameters = 3
    D_OUT = 2
elif dataset =='protein_trans_pathways':
    N_ODE_parameters = 6 
    D_OUT = 5

## Initialize the model
model = PyTorchDGP.DeepRegressionModel_ODE([1,2,D_OUT], sigma = sigma, l = l, init_noise = noise, init_Student_scale = -2, N_ODE_parameters = N_ODE_parameters)

## Optimization loop
start_time = time.time()
n_optimization_stages = 13
for optimization_stage in range(n_optimization_stages):
    Optimize.Optimize_ODE(x, y, model, dataset, optimization_stage, n_optimization_stages, D_OUT, Print=True)
stop_time = time.time()

## Save plot of ODE solution with parameters drawn from learned posterior 
misc.plot(x.data.numpy(),y.data.numpy(),model,coefs,initial_cond,tmin,tmax,dataset, path = 'RESULTS/' + dataset + '_ode_dgp_1dgf_EXPERIMENT_' + str(index_experiment) + '_FOLD_' + str(fold) + '_fit.png')                    

## Save results by saving some simple statistics of samples from the posterior over ODE parameters
L = model.branches[3].L_chol_cov_theta.data.numpy()
np.fill_diagonal(L, np.exp(model.branches[3].log_diag_L_chol_cov_theta.data.numpy()))
nsamples = 100
samples = []
for i in range(nsamples):
    ode_parameters =  np.exp(model.branches[3].m_theta.data.numpy() + np.matmul(L, np.random.randn(L.shape[0],1)))
    samples.append(ode_parameters)

## For the protein_trans_pathways dataset, we care about the ratio between V and km 
if dataset == 'protein_trans_pathways':
    samples = np.array(samples).squeeze() 
    samples = np.hstack([samples,(samples[:,4]/samples[:,5]).reshape([nsamples,1])])

percentiles = np.percentile(np.array(samples),[5,25,50,75,95],0).squeeze()

print("\nPercentiles (5th, 25th, 50th, 75th, 95th) of samples from the posterior over ODE parameters\n")
print(percentiles)
print("\nGround truth ", coefs, end='\n')

np.savetxt('RESULTS/' + dataset + '_ode_dgp_1dgf_EXPERIMENT_' + str(index_experiment) + '_FOLD_' + str(fold) + '_parameters.txt', percentiles)
np.savetxt('RESULTS/' + dataset + '_ode_dgp_1dgf_EXPERIMENT_' + str(index_experiment) + '_FOLD_' + str(fold) + '_elapsed.txt', [stop_time - start_time])
np.savetxt('RESULTS/' + dataset + '_ode_dgp_1dgf_EXPERIMENT_' + str(index_experiment) + '_FOLD_' + str(fold) + '_mean.txt', model.branches[3].m_theta.data.numpy())
np.savetxt('RESULTS/' + dataset + '_ode_dgp_1dgf_EXPERIMENT_' + str(index_experiment) + '_FOLD_' + str(fold) + '_sqrt-L.txt',L)
