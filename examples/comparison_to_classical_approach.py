# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 08:16:59 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import simulate

import numpy as np
import matplotlib.pyplot as plt

try:
    import parsimony.estimators as estimators
    import parsimony.utils.resampling as resampling
    import parsimony.algorithms.proximal as proximal
except ImportError:
    print("pylearn-parsimony is not properly installed. Will not be "
          "able to fit a model to the data.")


print("============================================")
print("=== Comparison to the classical approach ===")
print("============================================")

np.random.seed(42)
random_state = np.random.RandomState(42)
rng01 = simulate.utils.RandomUniform(0, 1, random_state=random_state)
rng_11 = simulate.utils.RandomUniform(-1, 1, random_state=random_state)

# Generate a start vector, beta_0, a candidate data set, X_0, and the
# residual vector, epsilon:
n, p = 50, 100
beta = simulate.beta.random((p + 1, 1), density=0.5, sort=True, rng=rng01)
Sigma = simulate.correlation_matrices.constant_correlation(p=p, rho=0.1,
                                                           eps=0.01,
                                                           random_state=random_state)
X0 = random_state.multivariate_normal(np.zeros(p), Sigma, n)
# Add a column of ones for the intercept
X0 = np.hstack((np.ones((n, 1)), X0))
e = 0.1 * random_state.randn(n, 1)

# Create the penalties:
lambda_l1 = 0.618
l1 = simulate.functions.L1(lambda_l1, rng=rng_11)
l2 = simulate.functions.L2Squared(1.0 - lambda_l1)

# Create the loss function:
lr = simulate.LinearRegressionData([l1, l2], X0, e, snr=5.0,
                                   intercept=True)

# Finally, generate the data:
X, y, beta_star, e = lr.load(beta)

# Precomputed start vector.
beta = [0.07615334, 0.01290186, 0.00202628, 0.02650592, 0.01641403,
        0.00832801, 0.00038979, 0.01853086, 0.02455442, 0.01755498,
        0.00469816, 0.0053463, -0.00264778, 0.0153526,  0.0015202,
        -0.02408666, 0.00453764, -0.0053824, -0.02556995, 0.01089147,
        -0.01863037, -0.00212071, -0.00307539, -0.00921482, 0.006925,
        0.00129344, 0.0281662, -0.00918977, -0.00449054, 0.0056015,
        0.02220001, 0.00913151, 0.00354939, 0.00156542, 0.00280743,
        0.03130712, 0.02208781, -0.01251621, 0.01020952, 0.00578535,
        0.00249149, 0.0176288, 0.00091225, -0.00582973, 0.01615972,
        0.01877612, -0.00849675, -0.0018213, -0.01855162, -0.01948095,
        0.02302791, 0.02071631, 0.03381445, 0.00185127, 0.00165616,
        0.00503767, 0.05038607, -0.01219906, 0.01042271, 0.01825756,
        0.02385574, 0.03347443, 0.00415551, 0.00362776, 0.02530247,
        -0.00496077, 0.02176268, -0.0286928, 0.0230842, 0.00629274,
        0.01643775, 0.03412733, 0.02470838, 0.04116392, 0.01187356,
        0.01273421, 0.00365497, 0.02987092, 0.01511648, -0.0004221,
        0.01879463, 0.04344603, 0.0272033, 0.01224449, 0.04655398,
        0.01646184, 0.01859317, 0.02223543, 0.01896585, 0.01484756,
        0.04559962, 0.00328105, 0.02131548, 0.02296882, 0.04799365,
        0.03204008, 0.01493607, 0.03915278, 0.01379136, 0.03394394,
        0.0341765]
beta = np.array(beta).reshape(p + 1, 1)

# Perform grid search
CV = []
ls = np.linspace(0.0, 1.0, num=100)
min_l = -1.0
min_e = np.float('inf')
for i in range(len(ls)):
    _l = ls[i]

    en = estimators.ElasticNet(_l,
                               algorithm=proximal.FISTA(),
                               algorithm_params=dict(max_iter=2000),
                               penalty_start=1, mean=False)

    score = []
    for tr, te in resampling.k_fold(n, K=10):

        # Training set
        Xtr = X[tr, :]
        ytr = y[tr, :]

        # Test set
        Xte = X[te, :]
        yte = y[te, :]

        # Fit a model.
        en.fit(Xtr, ytr, beta)

        # Compute and save prediction error.
        score.append(en.score(Xte, yte))

        # Reuse last beta for warm restart.
        beta = en.beta

    CV.append(np.mean(score))
    print "l1: %.3f, l2: %.3f, CV error: %.6f" % (_l, 1.0 - _l, CV[-1])

    if CV[-1] < min_e:
        min_e = CV[-1]
        min_l = _l

# Compute simulated (true) beta.
en = estimators.ElasticNet(lambda_l1,
                           algorithm=proximal.FISTA(),
                           algorithm_params=dict(max_iter=10000),
                           penalty_start=1, mean=False)
beta_sim = en.fit(X, y, beta).beta

# Compute the beta that corresponds to the CV minimum at min_l.
en = estimators.ElasticNet(min_l,
                           algorithm=proximal.FISTA(),
                           algorithm_params=dict(max_iter=10000),
                           penalty_start=1, mean=False)
beta_cv = en.fit(X, y, beta).beta

# Plot the results
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(ls, CV)
plt.xlabel(r"$\lambda_{\ell_1}$", fontsize=14)
plt.ylabel(r"CV error", fontsize=14)
plt.title(r"Cross-validated prediction error for different values of "
          r"$\lambda_{\ell_1}$", fontsize=18)

plt.subplot(2, 1, 2)
plt.plot(beta_star, '-g', linewidth=3)
plt.plot(beta_sim, '-b')
plt.plot(beta_cv, '-r')
plt.xlabel(r"$\beta_j$", fontsize=14)
plt.legend([r"$\beta^*$",
            r"$\beta_{Sim}$",
            r"$\beta_{CV}$"],
           loc=2)
plt.title(r"Regression vectors", fontsize=18)

plt.tight_layout()
plt.show()
