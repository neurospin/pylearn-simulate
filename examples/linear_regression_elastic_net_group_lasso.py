# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:26:19 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

import simulate

__all__ = ["lr_en_gl"]


def lr_en_gl():
    np.random.seed(42)

    test = False

    n, p = 48, 64 + 1
    groups = [range(1, 2 * p / 3), range(p / 3, p)]

    alpha = 1.0
    Sigma = alpha * np.eye(p - 1, p - 1) \
          + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
    mean = np.zeros(p - 1)
    M = np.random.multivariate_normal(mean, Sigma, n)
    M = np.hstack((np.ones((n, 1)), M))
    e = np.random.randn(n, 1)

    beta = np.random.rand(p - 1, 1)
    beta = np.sort(beta, axis=0)
    beta[0:(p - 1) / 2, :] = 0.0
    beta = np.vstack((np.random.rand(1, 1), beta))  # The intercept.

    snr = 100.0

    l = 0.618    # L1 coefficient
    k = 1.0 - l  # Ridge (L2) coefficient
    g = 1.618    # TV coefficient

    A = simulate.functions.GroupLasso.A_from_groups(p, groups, weights=None,
                                                    penalty_start=1)

    np.random.seed(42)
    penalties = [simulate.functions.L1(l),
                 simulate.functions.L2Squared(k),
                 simulate.functions.SmoothedGroupLasso(g, A,
                                                  mu=simulate.utils.TOLERANCE)]
    lr = simulate.LinearRegressionData(penalties, M, e, snr=snr,
                                       intercept=True)

    X, y, beta_star = lr.load(beta)

    try:
        import parsimony.estimators as estimators
        from parsimony.algorithms.primaldual import StaticCONESTA
        from parsimony.functions.combinedfunctions \
                import LinearRegressionL1L2GL
    except ImportError:
        print "pylearn-parsimony is not installed. Will not fit a model to " \
              "the data."
        return

    if test:
        max_iter = 5000
        n_vals = 3
        eps = 1e-5
    else:
        max_iter = 10000
        n_vals = 21
        eps = 1e-6

    ls = np.linspace(l - 0.25, l + 0.25, n_vals).tolist()
    gs = np.linspace(g - 0.25, g + 0.25, n_vals).tolist()

#    print "ls:", ls
#    print "gs:", gs

    beta = np.random.rand(p, 1)

    err_beta = np.zeros((n_vals, n_vals))
    err_f = np.zeros((n_vals, n_vals))

    l = ls[0]
    k = 1.0 - l
    g = gs[0]

    # Find a good starting point.
    lr = estimators.LinearRegressionL1L2GL(l1=l, l2=k, gl=g, A=A,
                                     algorithm=StaticCONESTA(max_iter=max_iter,
                                                             eps=eps),
                                     penalty_start=1, mean=False)
    beta = lr.fit(X, y, beta).beta

    for i in range(len(ls)):
        l = ls[i]
        k = 1.0 - l
        for j in range(len(gs)):
            g = gs[j]
#            print "l:", l, ", g:", g

            function = LinearRegressionL1L2GL(X, y, l, k, g, A=A,
                                              penalty_start=1, mean=False)

            lr = estimators.LinearRegressionL1L2GL(l1=l, l2=k, gl=g, A=A,
                                     algorithm=StaticCONESTA(max_iter=max_iter,
                                                             eps=eps),
                                     penalty_start=1, mean=False)
            beta = lr.fit(X, y, beta).beta

            err_beta[i, j] = np.linalg.norm(beta - beta_star)
            err_f[i, j] = np.linalg.norm(function.f(beta) \
                        - function.f(beta_star))

#            print err_beta
#            print err_f

#    print "err_beta:\n", err_beta
#    print "err_f:\n", err_f

    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plot
    import pylab

    np.random.seed(42)

    fig = plot.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(ls, gs)
    Z = err_f

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            antialiased=False, linewidth=0)
    #antialiased=False
    #ax.set_zlim(-1.01, 1.01)
    ax.patch.set_facecolor('none')
    ax.view_init(azim=-30, elev=30)

    plot.xlabel("$\lambda$", fontsize=14)
    plot.ylabel("$\gamma$", fontsize=14)
    plot.title(r"$f(\beta^{(k)}) - f(\beta^*)$", fontsize=16)

    x, y, _ = proj3d.proj_transform(l, g, np.min(Z), ax.get_proj())
    label = pylab.annotate(
        "$(0.618, 1.618)$", fontsize=14, xy=(x, y), xytext=(60, 20),
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

#    plot.savefig('lr_en_gl.pdf', bbox_inches='tight')
    plot.show()

if __name__ == "__main__":
    lr_en_gl()