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

import PyTorchDGP
import numpy as np
from numbers import Number
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


## The optimization strategy we used follows a series of steps where parameters are optimized in groups. 
## We found that initialization of the interpolant is important, so the first stage of the optimization is dedicated to that. 
## Next, we optimize sequentially (1) ODE parameters, (2) the interpolant, and (3) ODE parameters together with any parameters of the likelihood of the constraints
def Optimize_ODE(x, y, model, dataset, optimization_stage, n_optimization_stages, D_OUT, Print=True):

    print("********** Optimization stage", optimization_stage + 1, "of", n_optimization_stages, '  --  ', end='')

    if optimization_stage == 0:
        print("Optimizing interpolant and noise on observations")
        Niterations = 10000
        optimizer = torch.optim.Adam([{'params': model.branches[0].parameters(), 'lr': 1e-2}, \
                                      {'params': model.branches[1].parameters(), 'lr': 1e-2},\
                                      {'params': model.branches[2].parameters(), 'lr': 0},\
				      {'params': model.branches[3].parameters(), 'lr': 0 }])

    elif (optimization_stage -1)%3 == 0:
        print("Optimizing ODE parameters")
        Niterations = 8000
        optimizer = torch.optim.Adam([{'params': model.branches[0].parameters(), 'lr': 0}, \
                                      {'params': model.branches[1].parameters(), 'lr': 0},\
                                      {'params': model.branches[2].parameters(), 'lr': 0},\
                                      {'params': model.branches[3].parameters(), 'lr': 1e-2}])

    elif (optimization_stage - 2)%3 ==0:
        print("Optimizing Interpolant") 
        Niterations = 4000
        optimizer = torch.optim.Adam([{'params': model.branches[0].parameters(), 'lr': 1e-3}, \
                                      {'params': model.branches[1].parameters(), 'lr': 0},\
                                      {'params': model.branches[2].parameters(), 'lr': 0},\
                                      {'params': model.branches[3].parameters(), 'lr': 0}])


    elif (optimization_stage -3)%3==0: 
        print("Optimizing ODE parameters and Student t parameters")
        Niterations = 4000
        optimizer = torch.optim.Adam([{'params': model.branches[0].parameters(), 'lr': 0}, \
                                      {'params': model.branches[1].parameters(), 'lr': 0},\
                                      {'params': model.branches[2].parameters(), 'lr': 1e-3},\
                                      {'params': model.branches[3].parameters(), 'lr': 1e-3}])

    ## Optimization loop
    for iteration in range(Niterations+1):
        model.zero_grad()
        pred = model(x)

        ## Note again that in the first optimization stage we only optimize the interpolant
        if optimization_stage == 0:
            cost = -PyTorchDGP.Cost(y,pred)
        if optimization_stage > 0:
            cost = -PyTorchDGP.CostODE_StudentT(y, pred, pred, dataset)

        L = cost + model.KL()
        L.backward()
        optimizer.step()
        
        if Print:
        ## ********** Print
            if (iteration % 1000) == 0:
                print("\n** Iter = %8d" % iteration, "\t Round = %8d" %optimization_stage, "\tL = %8.3f" % L.data, end='') ## To be improved

                print("\n** Model hyper-parameters", end='')

                print("\nlog noise var =", end='')
                for i in range(D_OUT): 
                    print("\t%8.3f" % model.branches[1].sigma[i], end='')

                print("\nlog Student t scale =", end='')
                for i in range(D_OUT):
                    print("\t%8.3f" % model.branches[2].scale[i], end='')


                for i in range(model.nlayers):
                    print("\nlog lengthscale layer", i, "=", end='')
                    print("\t%8.3f" % model.branches[0][i].l.data[0], end='')

                for i in range(model.nlayers):
                    print("\nlog sigma layer", i, "=", end='')
                    print("\t%8.3f" % model.branches[0][i].sigma.data[0,0], end='')            

                print("\n** ODE parameters", end='')
                print("\nMeans =", end='')
                for i in range(len(model.branches[3].m_theta.data)): 
                    print("\t%8.3f" % np.exp(model.branches[3].m_theta.data[i]), end='')

                if not model.branches[3].factorized:
                    print("\nDiagonal covariance =", end='')
                    L = model.branches[3].L_chol_cov_theta.data.numpy()
                    np.fill_diagonal(L, np.exp(model.branches[3].log_diag_L_chol_cov_theta.data.numpy()))
                    print(np.diagonal(L))

                else:
                    print("\nVariances =", end='')
                    for i in range(len(model.branches[3].s_theta.data)): 
                        print("\t%8.3f" % np.exp(model.branches[3].s_theta.data[i]), end='')

                print("\n")
