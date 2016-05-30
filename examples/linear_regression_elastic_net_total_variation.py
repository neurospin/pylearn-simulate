# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:50:48 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

import simulate

__all__ = ["lr_en_tv"]


def lr_en_tv():
    print "======================================================================="
    print "=== Example with linear regression, elastic net and total variation ==="
    print "======================================================================="

    np.random.seed(42)

    test = False

    shape = (4, 4, 4)
    n, p = 48, np.prod(shape)

    # Generate candidate data.
    beta = simulate.beta.random((p, 1), density=0.5, sort=True)
    Sigma = simulate.correlation_matrices.constant_correlation(p=p, rho=0.01,
                                                               eps=0.001)
    X0 = np.random.multivariate_normal(np.zeros(p), Sigma, n)
    e = np.random.randn(n, 1)

    # Generate the linear operator for total variation.
    A = simulate.functions.TotalVariation.A_from_shape(shape)

    # Regularisation parameters
    k = 0.5  # Ridge (L2) coefficient.
    l = 1.0 - k  # L1 coefficient.
    g = 1.0  # TV coefficient.

    # Create the optimisation problem.
    np.random.seed(42)
    l1 = simulate.functions.L1(l)
    l2 = simulate.functions.L2Squared(k)
    tv = simulate.functions.TotalVariation(g, A)
    lr = simulate.LinearRegressionData([l1, l2, tv], X0, e, snr=3.0,
                                       intercept=False)

    # Generate simulated data.
    X, y, beta_star, e = lr.load(beta)

    # Define algorithm parameters.
    if test:
        max_iter = 5000
        n_vals = 3
        eps = 1e-5
        mu = 5e-6
    else:
        max_iter = 10000
        n_vals = 21
        eps = 5e-6
        mu = 5e-7

    try:
        import parsimony.estimators as estimators
        from parsimony.algorithms.proximal import CONESTA
        from parsimony.functions.combinedfunctions \
            import LinearRegressionL1L2TV
    except ImportError:
        print "pylearn-parsimony is not properly installed. Will not be " \
              "able to fit a model to the data."
        return

    ks = np.linspace(k - 0.25, k + 0.25, n_vals).tolist()
    gs = np.linspace(g - 0.25, g + 0.25, n_vals).tolist()

    # Precomputed start vector.
    beta = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -6.94158777e-08,
            -1.85184916e-09, 0.00000000e+00, 6.09145975e-09, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            4.68357405e-09, 2.04393050e-05, 8.09542282e-08, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -2.68424831e-07, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -3.62668134e-07, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            8.51958165e-07, 7.31258938e-07, 4.40944394e-07, 2.38028138e-03,
            2.53099740e-02, 1.33822793e-02, 6.79371496e-02, 1.04168020e-02,
            1.18304478e-06, 1.30844134e-06, 3.85488413e-03, 3.69889317e-02,
            8.91117508e-02, 5.86153319e-02, 8.51613851e-02, 8.93381453e-02,
            4.23168278e-02, 1.40616487e-01, 2.04455165e-01, 1.97578031e-01,
            2.64825748e-01, 2.64873032e-01, 2.69791537e-01, 2.71948706e-01,
            2.65233980e-01, 3.03929919e-01, 2.95572793e-01, 3.38838489e-01,
            3.43273009e-01, 3.80302127e-01, 4.35270115e-01, 4.37044532e-01]
    beta = np.array(beta).reshape(p, 1)

    err_beta = np.zeros((n_vals, n_vals))
    err_f = np.zeros((n_vals, n_vals))

    # Perform grid search.
    for i in range(len(ks)):
        k = ks[i]
        l = 1.0 - k
        for j in range(len(gs)):
            g = gs[j]
            print "k:", k, ", g:", g

            # Create the loss function.
            function = LinearRegressionL1L2TV(X, y, l, k, g, A=A, mu=mu,
                                              penalty_start=0, mean=False)

            # Create the estimator.
            lr = estimators.LinearRegressionL1L2TV(l, k, g, A=A, mu=mu,
                                     algorithm=CONESTA(max_iter=max_iter,
                                                       eps=eps),
                                     mean=False)
            # Fit data with the new regularisation parameters.
            beta = lr.fit(X, y, beta).beta

            # Compute output.
            err_beta[i, j] = np.linalg.norm(beta - beta_star)
            err_f[i, j] = np.linalg.norm(function.f(beta)
                                         - function.f(beta_star))

    print "err_beta:\n", err_beta
    print "err_f:\n", err_f

    # Plot the results.
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import pylab

    np.random.seed(42)

    # Plot results.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(ks, gs)
    Z = err_f

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           antialiased=False, linewidth=0)
    # antialiased=False
    # ax.set_zlim(-1.01, 1.01)
    ax.patch.set_facecolor('none')
    ax.view_init(azim=-45, elev=25)

    plt.xlabel("$\kappa$", fontsize=14)
    plt.ylabel("$\gamma$", fontsize=14)
    plt.title(r"$f(\beta^{(k)}) - f(\beta^*)$", fontsize=16)

    x, y, _ = proj3d.proj_transform(0.5, 1.0, np.min(Z), ax.get_proj())
    label = pylab.annotate(
        "$(0.5, 1.0)$", fontsize=14, xy=(x, y), xytext=(50, 20),
        textcoords='offset points', ha='right', va='bottom', color="white",
        # bbox=dict(boxstyle='round, pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0',
                        color="white"))

    def update_position(e):
        x, y, _ = proj3d.proj_transform(0.5, 1.0, np.min(Z), ax.get_proj())
        label.xy = x, y
        label.update_positions(fig.canvas.renderer)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_release_event', update_position)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf)  # , shrink=0.5, aspect=5)

#    plt.savefig('lr_en_tv.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    lr_en_tv()
