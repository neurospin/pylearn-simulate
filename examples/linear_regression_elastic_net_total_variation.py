# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:50:48 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

import simulate

import parsimony.estimators as estimators
import parsimony.algorithms.proximal as proximal
from parsimony.functions.combinedfunctions import LinearRegressionL1L2TV

np.random.seed(42)

shape = (4, 4, 4)
n, p = 48, np.prod(shape)

alpha = 1.0
Sigma = alpha * np.eye(p, p) \
      + (1.0 - alpha) * np.random.randn(p, p)
mean = np.zeros(p)
M = np.random.multivariate_normal(mean, Sigma, n)
e = np.random.randn(n, 1)
beta = np.random.rand(p, 1)
beta = np.sort(beta, axis=0)
beta[0:p / 2, :] = 0.0
snr = 100.0

l = 0.5  # L1 coefficient
k = 0.5  # Ridge coefficient
g = 1.0  # TV coefficient

A = simulate.functions.TotalVariation.A_from_shape(shape)

penalties = [simulate.functions.L1(l),
             simulate.functions.L2Squared(k),
             simulate.functions.TotalVariation(g, A)]
lr = simulate.LinearRegressionData(penalties, M, e, snr=snr, intercept=False)

X, y, beta_star = lr.load(beta)

max_iter = 20000
n_vals = 3
ks = np.linspace(0.25, 0.75, n_vals).tolist()
gs = np.linspace(0.75, 1.25, n_vals).tolist()
#mus = [0.5 ** x for x in range(25)]
mus = [0.1 ** x for x in range(9)]

print "ks:", ks
print "gs:", gs

beta = np.random.rand(p, 1)

err_beta = np.zeros((n_vals, n_vals))
err_f = np.zeros((n_vals, n_vals))

k = ks[0]
l = 1.0 - k
g = gs[0]
# Find a good starting point.
for mu in mus:
    print "mu:", mu
    lr = estimators.LinearRegressionL1L2TV(l1=l, l2=k, tv=g, A=A, mu=mu,
                                algorithm=proximal.FISTA(max_iter=max_iter,
                                                         eps=1e-5),
                                mean=False)
    beta = lr.fit(X, y, beta).beta

for i in range(len(ks)):
    k = ks[i]
    l = 1.0 - k
#    j = 1
    for j in range(len(gs)):
        g = gs[j]
        print "k:", k, ", g:", g
        for mu in mus:
            print "mu:", mu

            function = LinearRegressionL1L2TV(X, y, k, l, g, A=A,
                                              penalty_start=0, mean=False)
            eps = function.eps_opt(mu)

            lr = estimators.LinearRegressionL1L2TV(l1=l, l2=k, tv=g,
                                    A=A, mu=mu,
                                    algorithm=proximal.FISTA(max_iter=max_iter,
                                                             eps=eps),
                                    mean=False)
            beta = lr.fit(X, y, beta).beta

        err_beta[i, j] = np.linalg.norm(beta - beta_star)
        err_f[i, j] = np.linalg.norm(function.f(beta) - function.f(beta_star))

        print err_beta
        print err_f

print "err_beta:\n", err_beta
print "err_f:\n", err_f



#import numpy as np
#
#import simulate
#from simulate.correlation_matrices import constant_correlation
#
#np.random.seed(42)
#
#n, p = 64, 128
#
#mean = np.zeros(p)
#Sigma = constant_correlation(p=[p], rho=[0.05], delta=0.10, eps=0.5)
#X0 = np.random.multivariate_normal(mean, Sigma, n)
#e = np.random.randn(n, 1)
#
#beta_candidate = simulate.beta.random((p, 1), density=0.8,
#                                  rng=simulate.utils.RandomUniform(-1, 1).rand,
#                                  sort=True, normalise=False)
#
#l = 0.618
#k = 1.0 - l
#g = 1.618
#
#A = simulate.functions.TotalVariation.A_from_shape(p)
#
#snr = 20.0  # ~5 % noise
#
#penalties = [simulate.functions.L1(l),
#             simulate.functions.L1(k),
#             simulate.functions.TotalVariation(g, A)]
#lr = simulate.LinearRegressionData(penalties, X0, e, snr=snr, intercept=False)
#
#X, y, beta = lr.load(beta_candidate)