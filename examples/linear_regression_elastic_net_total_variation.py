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

    # Generate start values.
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

    l = 0.5  # L1 coefficient.
    k = 0.5  # Ridge (L2) coefficient.
    g = 1.0  # TV coefficient.

    # Create linear operator
    A = simulate.functions.TotalVariation.A_from_shape(shape)

    # Create optimisation problem.
    np.random.seed(42)
    penalties = [simulate.functions.L1(l),
                 simulate.functions.L2Squared(k),
                 simulate.functions.TotalVariation(g, A)]
    lr = simulate.LinearRegressionData(penalties, M, e, snr=snr,
                                       intercept=False)

    # Generate simulated data.
    X, y, beta_star = lr.load(beta)

    try:
        import parsimony.estimators as estimators
        from parsimony.algorithms.proximal import CONESTA
        from parsimony.functions.combinedfunctions \
                import LinearRegressionL1L2TV
    except ImportError:
        print "pylearn-parsimony is not properly installed. Will not fit a " \
              "model to the data."
        return

    if test:
        max_iter = 5000
        n_vals = 3
        eps = 1e-5
    else:
        max_iter = 10000
        n_vals = 21
        eps = 1e-6

    ks = np.linspace(0.25, 0.75, n_vals).tolist()
    gs = np.linspace(0.75, 1.25, n_vals).tolist()

#    print "ks:", ks
#    print "gs:", gs

    beta = np.random.rand(p, 1)

    err_beta = np.zeros((n_vals, n_vals))
    err_f = np.zeros((n_vals, n_vals))

    k = ks[0]
    l = 1.0 - k
    g = gs[0]

    # Find a good starting point.
    lr = estimators.LinearRegressionL1L2TV(l1=l, l2=k, tv=g, A=A,
                                     algorithm=CONESTA(max_iter=max_iter,
                                                       eps=eps),
                                     mean=False)
    beta = lr.fit(X, y, beta).beta

    # Perform grid search.
    for i in range(len(ks)):
        k = ks[i]
        l = 1.0 - k
        for j in range(len(gs)):
            g = gs[j]
            print "k:", k, ", g:", g

            function = LinearRegressionL1L2TV(X, y, k, l, g, A=A,
                                              penalty_start=0, mean=False)

            lr = estimators.LinearRegressionL1L2TV(l1=l, l2=k, tv=g, A=A,
                                     algorithm=CONESTA(max_iter=max_iter,
                                                       eps=eps),
                                     mean=False)
            beta = lr.fit(X, y, beta).beta

            err_beta[i, j] = np.linalg.norm(beta - beta_star)
            err_f[i, j] = np.linalg.norm(function.f(beta) \
                        - function.f(beta_star))

#            print err_beta
#            print err_f

    print "err_beta:\n", err_beta
    print "err_f:\n", err_f

    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plot
    import pylab

    np.random.seed(42)

    # Plot results.
    fig = plot.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(ks, gs)
    Z = err_f

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            antialiased=False, linewidth=0)
    #antialiased=False
    #ax.set_zlim(-1.01, 1.01)
    ax.patch.set_facecolor('none')
    ax.view_init(azim=-45, elev=25)

    plot.xlabel("$\kappa$", fontsize=14)
    plot.ylabel("$\gamma$", fontsize=14)
    plot.title(r"$f(\beta^{(k)}) - f(\beta^*)$", fontsize=16)

    x, y, _ = proj3d.proj_transform(0.5, 1.0, np.min(Z), ax.get_proj())
    label = pylab.annotate(
        "$(0.5, 1.0)$", fontsize=14, xy=(x, y), xytext=(50, 20),
        textcoords='offset points', ha='right', va='bottom', color="white",
    #    bbox=dict(boxstyle='round, pad=0.5', fc='white', alpha=0.5),
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

#    plot.savefig('lr_en_tv.pdf', bbox_inches='tight')
    plot.show()

if __name__ == "__main__":
    lr_en_tv()