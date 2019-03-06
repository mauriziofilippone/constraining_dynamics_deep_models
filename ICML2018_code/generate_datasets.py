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


## Code to generate NFOLDS datasets of size N for a given ODE model

import numpy as np
import matplotlib.pyplot as plt

import misc


class ExperimentConfiguration:
    
    def __init__(self, dataset, N, coefs, tmin, tmax, initial_conditions, noise_var):
        self.dataset = dataset
        self.N = N
        self.coefs = coefs
        self.tmin = tmin
        self.tmax = tmax
        self.initial_conditions = initial_conditions
        self.noise_var = noise_var

    def save(self, filename):

        np.savetxt(filename + '_N.txt', [self.N])
        np.savetxt(filename + '_coefs.txt', self.coefs)
        np.savetxt(filename + '_tmin.txt', [self.tmin])
        np.savetxt(filename + '_tmax.txt', [self.tmax])
        np.savetxt(filename + '_initial_conditions.txt', self.initial_conditions)
        np.savetxt(filename + '_noise_var.txt', [self.noise_var])



def generate_data(model,NFOLDS):
    ALL_EXPERIMENTS = []
    if model=='Lotka-Volterra':
        ## Two configs from ICML 2016 paper - one period and relatively smooth
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 34, [0.2,0.35,0.7,0.4], 0.0, 30.0, [1,2], 0.25**2))
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 51, [0.2,0.35,0.7,0.4], 0.0, 30.0, [1,2], 0.4**2))

        ## Four configs from preliminary runs - two periods
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 50, [0.5,0.02,0.1,0.037], 0.0, 100.0, [7,2], 1.0))
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 50, [0.5,0.02,0.1,0.037], 0.0, 100.0, [7,2], 10.0))
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 200, [0.5,0.02,0.1,0.037], 0.0, 100.0, [7,2], 1.0))
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 200, [0.5,0.02,0.1,0.037], 0.0, 100.0, [7,2], 10.0))

        ## Four configs from preliminary runs - three periods
        # ALL_EXPERIMENTS.append(ExperimentConfiguration("Lotka-Volterra", 50, [0.5,0.1,0.1,0.1], 0.0, 100.0, [7,2], 1.0))
        # ALL_EXPERIMENTS.append(ExperimentConfiguration("Lotka-Volterra", 50, [0.5,0.1,0.1,0.1], 0.0, 100.0, [7,2], 10.0))
        # ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 200, [0.5,0.1,0.1,0.1], 0.0, 100.0, [7,2], 0.5))
        # ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 500, [0.5,0.1,0.1,0.1], 0.0, 100.0, [7,2], 5.0))

    if model=='Fitz-Hugh-Nagumo':
        # Very noisy experiments
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 34, [1,0.2,0.7], 0.0, 200.0, [1,2], 0.25**2))
        #ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 34, [0.1,0.005,0.4], 0.0, 200.0, [1,2],0.1))

        # From: Campbell D, Steele RJ. Smooth functional tempering for nonlinear differential equation models. Stat Comput. 2012;22:429–43.
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 401, [3,0.2,0.2], 0.0, 20.0, [-1,1], 0.5**2))
        # Challenging variant from:  Benn Macdonald, Mu Niu, Simon Rogers, Maurizio Filippone and Dirk Husmeier. Approximate parameter inference in systems biology using gradient matching: a comparative evaluation. BioMedical Engineering OnLine
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 20, [3,0.2,0.2], 0.0, 10.0, [-1,1], 0.5**2))

    if model=="protein_trans_pathways":
        # From Macdonald et al
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 15, [0.07,0.6,0.05,0.3,0.017,0.3], 0.0, 100.0, [1,0,1,0,0], 0.1**2))
        ALL_EXPERIMENTS.append(ExperimentConfiguration(model, 15, [0.07,0.6,0.05,0.3,0.017,0.3], 0.0, 100.0, [1,0,1,0,0], 0.1**2/2))


    for index_experiment in range(len(ALL_EXPERIMENTS)):
        dataset = ALL_EXPERIMENTS[index_experiment].dataset
        N = ALL_EXPERIMENTS[index_experiment].N
        coefs = ALL_EXPERIMENTS[index_experiment].coefs
        tmin = ALL_EXPERIMENTS[index_experiment].tmin
        tmax = ALL_EXPERIMENTS[index_experiment].tmax
        initial_conditions = ALL_EXPERIMENTS[index_experiment].initial_conditions
        noise_var = ALL_EXPERIMENTS[index_experiment].noise_var

        filename_root = "FOLDS/" + dataset + '_EXPERIMENT_' + str(index_experiment)
        ALL_EXPERIMENTS[index_experiment].save(filename_root)    
    
        for fold in range(NFOLDS):

            DATA_X, DATA_Y = misc.generate_data(N, initial_conditions, coefs, dataset, tmin, tmax)
            DATA_Y = DATA_Y + np.sqrt(noise_var) * np.random.randn(DATA_Y.shape[0], DATA_Y.shape[1])
        
            filename_X = filename_root + '_FOLD_' + str(fold) + '_X.txt'
            filename_Y = filename_root + '_FOLD_' + str(fold) + '_Y.txt' 
        
            np.savetxt(filename_X, DATA_X)
            np.savetxt(filename_Y, DATA_Y)

        plt.plot(DATA_X, DATA_Y[:,0], "go")
        plt.plot(DATA_X, DATA_Y[:,1], "bo")

        plt.show()


## Generate all folds for all ODE models and parameter configurations
NFOLDS = 5
model = "Lotka-Volterra"
generate_data(model,NFOLDS)
model = "Fitz-Hugh-Nagumo"
generate_data(model,NFOLDS)
model = "protein_trans_pathways"
generate_data(model,NFOLDS)

