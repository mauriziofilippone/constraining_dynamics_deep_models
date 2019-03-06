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
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from misc import log_StudentT_likelihood
import torch.nn.functional as F

## Cost function for standard regression
def Cost(target, predicted):

    total_ll = 0

    dim = target.size(1)

    output_x = predicted[0]
    x = output_x[:,:dim]
    Dx = output_x[:,dim:]

    sigma = predicted[1]

    for i in range(dim):
        total_ll += -0.5 * (target.size(0)*(torch.log(sigma[i])) + torch.sum((target[:,i] - x[:,i]) ** 2)/(sigma[i]))

    return total_ll

## Cost function for regression over counts - Poisson likelihood
def Cost_Poisson(target, predicted):

    total_ll = 0

    dim = target.size(1)

    output_x = predicted[0]
    x = output_x[:,:dim]
    Dx = output_x[:,dim:]

    sigma = predicted[1]
    for i in range(dim):
        mu = torch.exp(x[:,i])
        total_ll +=  torch.sum(torch.log(mu) * target[:,i] - mu - (target[:,i]+1)*(torch.log(target[:,i])))  
    
    return total_ll


## Cost function for regression with student T likelihood for the constraint 
## In our experiments we use the student T with a large degree of freedom to obtain the effect of the Gaussian likelihood for the constraint
def CostODE_StudentT(target, predicted, predicted2, model):

    dim = target.size(1)

    output_x = predicted[0]
    x = output_x[:,:dim]
    Dx = output_x[:,dim:]

    sigma = predicted[1]
    scale = predicted[2]
    output_theta = predicted[3]

    df = Variable(torch.Tensor([[1.]]))

    total_ll = 0

    if not model=="monotonic":
        for i in range(dim):
            total_ll += -0.5 * (target.size(0)*(torch.log(sigma[i])) + torch.sum((target[:,i] - x[:,i]) ** 2)/(sigma[i]))
    else:
        total_ll = Cost_Poisson(target,predicted)

    output_x2 = predicted2[0]
    x2 = output_x2[:,:dim]
    Dx2 = output_x2[:,dim:]

    scale = predicted2[2]
    output_theta2 = predicted2[3]

    if model=='Lotka-Volterra':
        a,b,c,d = output_theta
        constraint1 = log_StudentT_likelihood(Dx[:,0], torch.exp(a) * x[:,0] - torch.exp(b) * x[:,0] * x[:,1], scale[0], df )
        constraint2 = log_StudentT_likelihood(Dx[:,1], -torch.exp(c) * x[:,1] + torch.exp(d) * x[:,0] * x[:,1], scale[1], df )
        constraint = constraint1 + constraint2

    if model=='Fitz-Hugh-Nagumo':
        a,b,c = output_theta2
        constraint1 = log_StudentT_likelihood(Dx2[:,0],  torch.exp(a) * (x2[:,0] - torch.pow(x2[:,0], 3) / 3. + x2[:,1]),scale[0],df)
        constraint2 = log_StudentT_likelihood(Dx2[:,1], -torch.exp(-a) * (x2[:,0] - torch.exp(b) + torch.exp(c) * x2[:,1]),scale[1],df)
        constraint = constraint1 + constraint2

    if model=='protein_trans_pathways':
        theta0,theta1,theta2,theta3,theta4,theta5 = predicted2[3]
        constraint1 = log_StudentT_likelihood(Dx2[:,0], -torch.exp(theta0) * x2[:,0] - torch.exp(theta1) * x2[:,0] * x2[:,2] + torch.exp(theta2) * x2[:,3], scale[0],df)
        constraint2 = log_StudentT_likelihood(Dx2[:,1],  torch.exp(theta0) * x2[:,0], scale[1],df)
        constraint3 = log_StudentT_likelihood(Dx2[:,2], -torch.exp(theta1) * x2[:,0] * x2[:,2] + torch.exp(theta2) * x2[:,3] + torch.exp(theta4) * x2[:,4]/(torch.exp(theta5) + x2[:,4]), scale[2],df)
        constraint4 = log_StudentT_likelihood(Dx2[:,3], torch.exp(theta1) * x2[:,0] * x2[:,2] - torch.exp(theta2) * x2[:,3] - torch.exp(theta3) * x2[:,3], scale[3],df)
        constraint5 = log_StudentT_likelihood(Dx2[:,4], torch.exp(theta3) * x2[:,3] - torch.exp(theta4) * x2[:,4] / (torch.exp(theta5) + x2[:,4]), scale[4],df)
        constraint = constraint1 + constraint2 + constraint3 + constraint4 + constraint5


    if model=='monotonic':
        theta0 = predicted2[3]
        x_relu = F.relu(-theta0 * Dx2)        
        constraint = -torch.sum(x_relu)

    return total_ll + constraint

## This is the main model class
## It is constructed as a nn "Sequential" module, where model components are added as branches
class DeepRegressionModel_ODE(nn.Module):
    def __init__(self, sizes, N_rf = 20 , seed = np.int(1), l = 0.1, sigma = 1 , init_noise = 3.0, N_ODE_parameters = 4, init_Student_scale = 0, init_param_ODE = -2):
        super(DeepRegressionModel_ODE, self).__init__()
        
        self.nlayers = len(sizes) - 1

        GP_list = [GP(sizes[i], sizes[i+1], N_rf, seed, l, sigma, level = 1) for i in range(len(sizes)-1)]
        GP_list[0].level = 0
        DGP = nn.Sequential(OrderedDict([('GP'+str(i),GP_list[i]) for i in range(len(sizes)-1)]))
        self.branches = nn.ModuleList([DGP, Gauss_NoiseModel(sizes[len(sizes)-1], init_noise), StudentT_NoiseModel(sizes[len(sizes)-1],init_Student_scale), ODE_parameters(N_ODE_parameters, 1, init_param_ODE)])

        for i, branch in enumerate(self.branches):
            self.add_module(str(i), branch)

    def forward(self, x):
        return self.branches[0](x), self.branches[1](x), self.branches[2](x), self.branches[3](x)

    def KL(self):
        KL_tot = 0
        for i in range(len(self.branches[0])):
            KL_tot += torch.sum(self.branches[0][i].KL())
        return KL_tot + self.branches[3].KL()

## One model branch contains the mean and (a decomposition of the) covariance of the approximate posterior over ODE parameters
## In the factorized case, the decomposition of the covariance is replaced by the logarithm of the variances
class ODE_parameters(nn.Module):

    def __init__(self, N_parameters, N_output_dim, init_param_ODE = -2):

        super(ODE_parameters, self).__init__()

        ## We hard-code the non-factorization of the posterior over theta as this is what we use in all the experiments
        self.factorized = False

        self.N_parameters = N_parameters
        self.N_output_dim = N_output_dim

        self.m_theta = nn.Parameter(torch.Tensor(self.N_parameters, self.N_output_dim).fill_(init_param_ODE), requires_grad=True)
        if self.factorized:
            self.s_theta = nn.Parameter(torch.Tensor(self.N_parameters, self.N_output_dim).fill_(-3), requires_grad=True)
        
        ## In the non-factorized case, we optimize a lower triangular decomposition L of the covariance of the posterior over ODE parameters (Sigma = L L')
        ## Note that we parameterize the diagonal of L to work with logarithms so as to ensure its positiveness
        if not self.factorized:
            self.L_chol_cov_theta = nn.Parameter(torch.zeros(self.N_parameters, self.N_parameters), requires_grad=True)
            self.log_diag_L_chol_cov_theta = nn.Parameter(torch.zeros(self.N_parameters), requires_grad=True)

    ## Here we draw samples from the approximate posterior - differently depending on whether the posterior is factorized or not
    def forward(self,x):

        sampler = Variable(torch.randn(self.N_parameters, self.N_output_dim), requires_grad=False).type(torch.FloatTensor)
        if self.factorized:
            exp_s_theta = torch.exp(self.s_theta)
            theta = torch.sqrt(exp_s_theta)*sampler + self.m_theta

        if not self.factorized:
            self.L_chol_cov_theta.data = torch.tril(self.L_chol_cov_theta.data)
            self.L_chol_cov_theta.data -= torch.diag(torch.diag(self.L_chol_cov_theta.data))
            theta = torch.add(torch.matmul(self.L_chol_cov_theta + torch.diag(torch.exp(self.log_diag_L_chol_cov_theta)), sampler), self.m_theta)

        return theta

    ## The KL term pertaining to ODE parameters in the variational formulation - differently depending on whether the posterior is factorized or not
    def KL(self):

        if self.factorized:
            return 0.5*torch.sum(torch.exp(self.s_theta) + self.m_theta**2 - torch.log(torch.exp(self.s_theta)) - 1)

        if not self.factorized:
            self.L_chol_cov_theta.data = torch.tril(self.L_chol_cov_theta.data)
            self.L_chol_cov_theta.data -= torch.diag(torch.diag(self.L_chol_cov_theta.data))

            dimension = self.m_theta.shape[0]

            return 0.5 * (- 2.0 * torch.sum(torch.log(torch.diag(self.L_chol_cov_theta + torch.diag(torch.exp(self.log_diag_L_chol_cov_theta))))) +
                          torch.sum(torch.pow(self.m_theta, 2)) + 
                          torch.sum(torch.diag(torch.matmul(self.L_chol_cov_theta + torch.diag(torch.exp(self.log_diag_L_chol_cov_theta)), (self.L_chol_cov_theta + torch.diag(torch.exp(self.log_diag_L_chol_cov_theta))).t()))) - dimension)

## Student T likelihood for the constraint 
class StudentT_NoiseModel(nn.Module):
    def __init__(self, out_sizeStudentT, init_scale):
        super(StudentT_NoiseModel, self).__init__()  # always call parent's init
        self.scale = nn.Parameter(torch.Tensor(out_sizeStudentT).fill_(init_scale), requires_grad=True)
    def forward(self, x):
        return torch.exp(self.scale)


## Gaussian likelihood for the constraint 
class Gauss_NoiseModel(nn.Module):
    def __init__(self, out_size, init):
        super(Gauss_NoiseModel, self).__init__()  # always call parent's init
        self.sigma = nn.Parameter(torch.Tensor(out_size).fill_(init), requires_grad=True)
    def forward(self, x):
        return torch.exp(self.sigma)

## Gaussian process module with Random Feature Expansion approximation
class GP(nn.Module):
    def __init__(self, input_dim, output_dim, N_rf, seed, l, sigma, prior_fixed=True, var_fixed=False, order = 1, level = 0):
        super(GP, self).__init__()  # always call parent's init

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N_rf = N_rf
        self.prior_fixed = prior_fixed

        self.seed = seed
        self.order = order
        self.level = level

        ## Covariance parameters
        self.l = nn.Parameter(torch.log(torch.Tensor(input_dim).fill_(l)), requires_grad=True)
        self.sigma = nn.Parameter(torch.Tensor([[sigma]]), requires_grad=True)

        ## Variational parameters over the frequencies 
        self.m_omega = nn.Parameter(torch.Tensor(input_dim, N_rf), requires_grad=True)
        self.s_omega = nn.Parameter(torch.Tensor(input_dim, N_rf), requires_grad=True)
        self.m_omega.data = torch.rand(input_dim, N_rf) - torch.FloatTensor([[0.5]])
        self.s_omega.data = torch.rand(input_dim, N_rf) - torch.FloatTensor([[0.5]])

        ## Variational parameters over the coefficients of the linear combination of random features 
        self.m_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, output_dim), requires_grad=True)
        self.s_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, output_dim), requires_grad=True)
        self.m_w.data = 2 * torch.rand(2*self.N_rf + 1, output_dim) - torch.FloatTensor([[1.]])
        self.s_w.data = torch.rand(2*self.N_rf + 1, output_dim) - torch.FloatTensor([[0.5]])

    def forward(self, x):
            if self.order == 1:
                if self.level > 0:
                    input_x = x[:, :self.input_dim]
                    Dinput_x = x[:, self.input_dim:]
                else:
                    ## Assuming time as input
                    input_x = x
                    Dinput_x = Variable(torch.ones(x.size(0), x.size(1)), requires_grad=False).type(torch.FloatTensor)

            ## With the prior_fixed option, the frequencies Omega are not learned variationally but are sampled from the prior
            ## Note that this is the only option implemented - check ICML 2017 paper on DGPs for other possible options on how to treat Omega
            if self.prior_fixed:
                exp_l = 1 / torch.exp(self.l)
                exp_sigma = torch.exp(self.sigma)
                rng = np.random.RandomState(self.seed)
                sampler = Variable(torch.from_numpy(rng.randn(self.input_dim, self.N_rf)), requires_grad=False).type(
                    torch.FloatTensor)
                omega = Variable(torch.zeros(self.input_dim, self.N_rf))
                for p in range(len(self.l)):
                    omega[p] = torch.sqrt(exp_l[p]) * sampler[p]
                #omega = torch.sqrt(torch.exp(self.s_omega)) * sampler + self.m_omega
                sqrt_len_omega = Variable(torch.sqrt(torch.FloatTensor([self.N_rf])))

                ## Order determines the order of derivatives needed in the computations
                ## If order==1 then we construct also the random features needed to obtain the derivative of the GP latent function
                if self.order == 0:
                    Phi = (torch.sqrt(exp_sigma) / sqrt_len_omega) * torch.cat(
                            [torch.cos(torch.mm(x, omega)), torch.sin(torch.mm(x, omega))], dim=1)
                elif self.order == 1:
                    Phi = (torch.sqrt(exp_sigma) / sqrt_len_omega) * \
                               torch.cat([torch.cos(torch.mm(input_x, omega)), torch.sin(torch.mm(input_x, omega)),
                                          -torch.mul(torch.sin(torch.mm(input_x, omega)), torch.mm(Dinput_x, omega)),
                                          torch.mul(torch.cos(torch.mm(input_x, omega)), torch.mm(Dinput_x, omega))],
                                         dim=1)

            exp_s_w = torch.exp(self.s_w)
            sampler = Variable(torch.randn(2*self.N_rf + 1, self.output_dim), requires_grad=False).type(torch.FloatTensor)
            W = torch.sqrt(exp_s_w) * sampler + self.m_w

            if self.order == 0:
                return torch.mm(Phi, W[1:,:]) + W[0,:]
            elif self.order == 1:
                product = torch.mm(Phi[:, : 2*self.N_rf], W[1:,:])
                Dproduct = torch.mm(Phi[:, 2*self.N_rf:], W[1:,:])
                return torch.cat([product, Dproduct], dim=1)


    ## KL term for W parameters
    def KL(self):
        KL = 0.5*torch.sum(torch.exp(self.s_w) + self.m_w**2 - torch.log(torch.exp(self.s_w)) - 1)
        return(KL)
