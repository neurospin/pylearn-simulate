# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:49:02 2016

Copyright (c) 2013-2016, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest

import numpy as np

from tests import TestCase
import simulate.utils as utils
import simulate


class TestSimulate(TestCase):

    def test_lasso(self):

        random_state = np.random.RandomState(42)
        rng01 = utils.RandomUniform(0, 1, random_state=random_state)
        rng11 = utils.RandomUniform(-1, 1, random_state=random_state)

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
        l = 0.618
        l1 = simulate.functions.L1(l, rng=rng11)
#        l2 = simulate.functions.L2Squared(1.0 - l)

        # Create the loss function:
        lr = simulate.LinearRegressionData([l1], X0, e, snr=5.0,
                                           intercept=True)

        # Finally, generate the data:
        X, y, beta_star, e = lr.load(beta)

        betanew = betaold = beta
        for i in range(10000):

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
            step = 0.00002
            betaold = betanew
            betanew = z - step * X.T.dot(X.dot(z) - y)
            l_ = l * step
            x = betanew[1:, :]
            x = (np.abs(x) > l_) * (x - l_ * np.sign(x - l_))
            betanew[1:, :] = x

        assert(np.linalg.norm(betanew - beta_star) < 0.01)

    def test_elastic_net(self):

        random_state = np.random.RandomState(42)
        rng01 = utils.RandomUniform(0, 1, random_state=random_state)
        rng11 = utils.RandomUniform(-1, 1, random_state=random_state)

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
        l = 0.618
        l1 = simulate.functions.L1(l, rng=rng11)
        l2 = simulate.functions.L2Squared(1.0 - l)

        # Create the loss function:
        lr = simulate.LinearRegressionData([l1, l2], X0, e, snr=5.0,
                                           intercept=True)

        # Finally, generate the data:
        X, y, beta_star, e = lr.load(beta)

        betanew = betaold = beta
        for i in range(10000):

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
            step = 0.00002
            betaold = betanew
            betanew = z - step * (X.T.dot(X.dot(z) - y) + (1.0 - l) * z)
            l_ = l * step
            x = betanew[1:, :]
            x = (np.abs(x) > l_) * (x - l_ * np.sign(x - l_))
            betanew[1:, :] = x

        assert(np.linalg.norm(betanew - beta_star) < 0.005)

    def test_total_variation(self):

        random_state = np.random.RandomState(42)
        state = random_state.get_state()
        rng01 = simulate.utils.RandomUniform(0, 1, random_state=random_state)
        rng11 = simulate.utils.RandomUniform(-1, 1, random_state=random_state)

        shape = (4, 4, 4)
        n, p = 48, np.prod(shape)

        # Generate candidate data.
        beta = simulate.beta.random((p, 1), density=0.5, sort=True, rng=rng01)
        Sigma = simulate.correlation_matrices.constant_correlation(p=p,
                                                                   rho=0.01,
                                                                   eps=0.001,
                                                                   random_state=random_state)
        X0 = random_state.multivariate_normal(np.zeros(p), Sigma, n)
        e = random_state.randn(n, 1)

        # Generate the linear operator for total variation.
        A = simulate.functions.TotalVariation.A_from_shape(shape)
    
        # Regularisation parameters
        k = 0.5  # Ridge (L2) coefficient.
        l = 1.0 - k  # L1 coefficient.
        g = 1.0  # TV coefficient.
        mu = 5e-4
    
        # Create the optimisation problem.
        random_state.set_state(state)
        l1 = simulate.functions.L1(l, rng=rng11)
        l2 = simulate.functions.L2Squared(k)
        tv = simulate.functions.TotalVariation(g, A, rng=rng01)
        lr = simulate.LinearRegressionData([l1, l2, tv], X0, e, snr=3.0,
                                           intercept=False)

        # Generate simulated data.
        X, y, beta_star, e = lr.load(beta)

        try:
            import parsimony.estimators as estimators
            import parsimony.algorithms.proximal as proximal
        except ImportError:
            print "pylearn-parsimony is not properly installed. Will not be " \
                  "able to run this test."
            return

        e = estimators.LinearRegressionL1L2TV(l, k, g, A, mu,
                                              algorithm=proximal.FISTA(),
                                              algorithm_params=dict(max_iter=10000),
                                              penalty_start=0, mean=False)
        beta_sim = e.fit(X, y, beta).parameters()["beta"]

        assert(np.linalg.norm(beta_sim - beta_star) < 0.0005)

    def test_group_lasso(self):

        random_state = np.random.RandomState(42)
        state = random_state.get_state()
        rng01 = simulate.utils.RandomUniform(0, 1, random_state=random_state)
        rng11 = simulate.utils.RandomUniform(-1, 1, random_state=random_state)

        # Generate start values.
        n, p = 48, 64 + 1

        # Define the groups.
        groups = [range(1, 2 * p / 3), range(p / 3, p)]

        # Generate candidate data.
        beta = simulate.beta.random((p - 1, 1), density=0.5, sort=True,
                                    rng=rng01)
        # Add the intercept.
        beta = np.vstack((random_state.rand(1, 1), beta))
        Sigma = simulate.correlation_matrices.constant_correlation(p=p - 1,
                                                                   rho=0.01,
                                                                   eps=0.001,
                                                                   random_state=random_state)
        X0 = random_state.multivariate_normal(np.zeros(p - 1), Sigma, n)
        # Add the intercept.
        X0 = np.hstack((np.ones((n, 1)), X0))
        e = random_state.randn(n, 1)

        # Create linear operator.
        A = simulate.functions.SmoothedGroupLasso.A_from_groups(p, groups,
                                                                weights=None,
                                                                penalty_start=1)

        # Define regularisation parameters.
        l = 0.618    # L1 coefficient.
        k = 1.0 - l  # Ridge (L2) coefficient.
        g = 1.618    # TV coefficient.
        mu = 5e-4

        # Create optimisation problem.
        l1 = simulate.functions.L1(l, rng=rng11)
        l2 = simulate.functions.L2Squared(k)
        gl = simulate.functions.SmoothedGroupLasso(g, A,
                                                   mu=simulate.utils.TOLERANCE)
        lr = simulate.LinearRegressionData([l1, l2, gl], X0, e, snr=2.0,
                                           intercept=True)

        # Generate simulated data.
        random_state.set_state(state)
        X, y, beta_star, e = lr.load(beta)

        try:
            import parsimony.estimators as estimators
            import parsimony.algorithms.proximal as proximal
        except ImportError:
            print "pylearn-parsimony is not properly installed. Will not be " \
                  "able to run this test."
            return

        e = estimators.LinearRegressionL1L2GL(l, k, g, A, mu,
                                              algorithm=proximal.FISTA(max_iter=10000),
                                              penalty_start=1, mean=False)
        beta_sim = e.fit(X, y, beta).parameters()["beta"]

        assert(np.linalg.norm(beta_sim - beta_star) < 0.0001)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()
