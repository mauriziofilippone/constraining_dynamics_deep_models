# README #

This repository contains code to reproduce the results in the paper:

[1] M. Lorenzi and M. Filippone. Constraining the Dynamics of Deep Probabilistic Models. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, 2018.

The code is written in python and uses the pytorch module. Our code has been tested with python 3.7.

## Flags ##

The code implements variational inference for deep Gaussian processs approximated using random Fourier features with various constraints. 

## Examples ##

Here are a few examples on how to run the code

### Regression with ODE-based constraints ###

```
#!bash
# Here is an example where we run our mode on the Lotka-Volterra ODE (fold 0 of experimental setup 0)

python3 test_ODE.py Lotka-Volterra 0 0

```

```
#!bash
# Similarly, it is possible to run the code for other ODE systems and for other fold/experimental setup configurations

python3 test_ODE.py Fitz-Hugh-Nagumo 2 1

python3 test_ODE.py protein_trans_pathways 3 0 

```
