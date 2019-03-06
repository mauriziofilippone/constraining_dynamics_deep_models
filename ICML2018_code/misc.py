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

import numpy as np
from numbers import Number
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def standardize(X):
    return 2*(X - np.min(X)) / (np.max(X) - np.min(X)) - 1, np.min(X), np.max(X)

## Lotka-Volterra 
def ode_lotka_interpolant(x, t, log_theta):
    dx1 = np.exp(log_theta[0]) * x[:,0] - np.exp(log_theta[1]) * x[:,0] * x[:,1]
    dx2 = -np.exp(log_theta[2]) * x[:,1] + np.exp(log_theta[3]) * x[:,0] * x[:,1]
    return np.stack([dx1,dx2]).T.squeeze()
 
## Fitz-Hugh-Nagumo
def ode_fitz_interpolant(x, log_theta):
    dx1 = np.exp(log_theta[0]) * (x[:,0] - np.pow(x[:,0], 3) / 3. + x[:,1])
    dx2 = -np.exp(-log_theta[0]) * (x[:,0] - np.exp(log_theta[1]) + np.exp(log_theta[2]) * x[:,1])
    return np.stack([dx1,dx2]).T.squeeze()

## Protein pathways
def protein_trans_pathways(x, log_theta):
    # x[:,0] = S
    # x[:,1] = dS
    # x[:,2] = R
    # x[:,3] = RS
    # x[:,4] = Rpp
    # log_theta -> 0:k1, 1:k2, 2:k3, 3:k4, 4:V, 5:Km
    dx1 = -np.exp(log_theta[0]) * x[:,0] - np.exp(log_theta[1]) * x[:,0] * x[:,2] + np.exp(log_theta[2]) * x[:,3]
    dx2 =  np.exp(log_theta[0]) * x[:,0]
    dx3 = -np.exp(log_theta[1]) * x[:,0] * x[:,2] + np.exp(log_theta[2]) * x[:,3] + np.exp(log_theta[4]) * x[:,4]/(np.exp(log_theta[5]) + x[:,4])
    dx4 = np.exp(log_theta[1]) * x[:,0] * x[:,2] - np.exp(log_theta[2]) * x[:,3] - np.exp(log_theta[3]) * x[:,3]
    dx5 = np.exp(log_theta[3]) * x[:,3] - np.exp(log_theta[4]) * x[:,4] / (np.exp(log_theta[5]) + x[:,4])

    return np.stack([dx1,dx2,dx3,dx4,dx5]).T.squeeze()


## Data generation
from scipy.integrate import odeint

def generate_data(n, x0, theta0, dataset, tmin, tmax): 
    if dataset == "Lotka-Volterra":    
        def ode_lotka_volterra(x, t, theta):
            return [theta[0] * x[0] - theta[1] * x[0] * x[1], -theta[2] * x[1] + theta[3] * x[0] * x[1]]

        t = np.linspace(tmin, tmax, n)
        y = odeint(ode_lotka_volterra, x0, t, args=(theta0,))

    if dataset == "Fitz-Hugh-Nagumo":    
        def ode_fitz(x, t, theta):
            return [theta[0] * (x[0] - np.power(x[0], 3) / 3. + x[1]), -(x[0] - theta[1] + theta[2] * x[1]) / theta[0]]

        t = np.linspace(tmin, tmax, n)
        y = odeint(ode_fitz, x0, t, args=(theta0,))

    if dataset == "protein_trans_pathways":
        def ode_protein(x, t, theta):
            return [-theta[0] * x[0] - theta[1] * x[0] * x[2] + theta[2] * x[3] ,\
                     theta[0] * x[0],\
                    -theta[1] * x[0] * x[2] + theta[2] * x[3] + theta[4] * x[4]/(theta[5] + x[4]),\
                     theta[1] * x[0] * x[2] - theta[2] * x[3] - theta[3] * x[3],\
                     theta[3] * x[3] - theta[4] * x[4]/(theta[5] + x[4])   ]

        t = np.array([0,1,2,4,5,7,10,15,20,30,40,50,60,80,100]) 
        y = odeint(ode_protein, x0, t, args=(theta0,))

    return t, y

def plot(X,Y,model,coefs,initial_conditions,tmin,tmax,dataset,path = ''):
    x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    for i in range(100):
        prediction = model(x)[0][:, :Y.shape[1]].data.numpy()
        for dim in range(Y.shape[1]):
            col = cm.tab10(dim)
            ax.plot(X, prediction[:, dim], color=col, alpha=0.1)

    for dim in range(Y.shape[1]):
        col = cm.tab10(dim)
        ax.scatter(X, Y[:, dim], color=col, alpha=0.1)

    real_x,real_y = generate_data(500, initial_conditions, coefs.flatten(), dataset, tmin, tmax)
    for dim in range(Y.shape[1]):
        col = cm.tab10(dim)
        ax.scatter(real_x, real_y[:, dim], color=col)    

    if len(path) > 0:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    else:
        fig.show()
        plt.close(fig)

def read_data(filename_root, fold):
    #N = np.loadtxt(filename_root + '_N.txt')
    coefs = np.loadtxt(filename_root + '_coefs.txt')
    tmin = np.loadtxt(filename_root + '_tmin.txt')
    tmax = np.loadtxt(filename_root + '_tmax.txt')
    initial_conditions = np.loadtxt(filename_root + '_initial_conditions.txt')
    noise_var = np.loadtxt(filename_root + '_noise_var.txt')

    filename_X = filename_root + '_FOLD_' + str(fold) + '_X.txt'
    filename_Y = filename_root + '_FOLD_' + str(fold) + '_Y.txt'

    X = np.loadtxt(filename_X, delimiter=' ')
    X = X.reshape([len(X),1])
    Y = np.loadtxt(filename_Y, delimiter=' ')

    x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)

    return x,y,coefs,initial_conditions,tmin,tmax

def read_data_monotonic(path):
    data_input =  pd.read_csv(path,sep=';')
    x = np.array(data_input.age)
    y = np.array(data_input.y)
    out_y = y/np.array(data_input.N) *1000
    return Variable(torch.from_numpy(x), requires_grad=False).type(torch.FloatTensor).resize(len(x),1), Variable(torch.from_numpy(y), requires_grad=False).type(torch.FloatTensor).resize(len(y),1)


def log_StudentT_likelihood(target, predicted, scale, df):
    y = (target - predicted) 
    Z = (-0.5 * scale.log() +
             0.5 * df.log() +
             0.5 * math.log(math.pi) +
             torch.lgamma(0.5 * df) -
             torch.lgamma(0.5 * (df + 1.)))
    return -0.5 * (df + 1.) * torch.sum(torch.log1p(scale**2 * y**2. / df)) - target.size(0) * Z


def log_StudentT_likelihood_tensor(target, predicted, scale, df):
    y = (target - predicted) 
    Z = torch.sum(-0.5 * scale.log() +
             0.5 * df.log() +
             0.5 * math.log(math.pi) +
             torch.lgamma(0.5 * df) -
             torch.lgamma(0.5 * (df + 1.)))
    return torch.sum(-0.5 * (df + 1.) * (torch.log1p(scale**2 * y**2. / df))) - target.size(0) * Z

