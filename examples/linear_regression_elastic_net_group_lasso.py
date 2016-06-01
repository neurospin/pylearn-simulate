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
    print "==================================================================="
    print "=== Example with linear regression, elastic net and group lasso ==="
    print "==================================================================="

#    np.random.seed(42)
    random_state = np.random.RandomState(42)
    state = random_state.get_state()
    rng01 = simulate.utils.RandomUniform(0, 1, random_state=random_state)
    rng_11 = simulate.utils.RandomUniform(-1, 1, random_state=random_state)

    test = False

    # Generate start values.
    n, p = 48, 64 + 1

    # Define the groups.
    groups = [range(1, 2 * p / 3), range(p / 3, p)]

    # Generate candidate data.
    beta = simulate.beta.random((p - 1, 1), density=0.5, sort=True, rng=rng01)
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

    # Create optimisation problem.
    l1 = simulate.functions.L1(l, rng=rng_11)
    l2 = simulate.functions.L2Squared(k)
    gl = simulate.functions.SmoothedGroupLasso(g, A,
                                               mu=simulate.utils.TOLERANCE)
    lr = simulate.LinearRegressionData([l1, l2, gl], X0, e, snr=2.0,
                                       intercept=True)

    # Generate simulated data.
#    np.random.seed(42)
    random_state.set_state(state)
    X, y, beta_star, e = lr.load(beta)

    # Define algorithm parameters.
    if test:
        max_iter = 5000
        n_vals = 3
        eps = 1e-5
    else:
        max_iter = 10000
        n_vals = 21
        eps = 1e-6

    try:
        import parsimony.estimators as estimators
        from parsimony.algorithms.proximal import CONESTA
        from parsimony.functions.combinedfunctions \
            import LinearRegressionL1L2GL
    except ImportError:
        print "pylearn-parsimony is not properly installed. Will not be " \
              "able to fit a model to the data."
        return

    ls = np.linspace(l - 0.25, l + 0.25, n_vals).tolist()
    gs = np.linspace(g - 0.25, g + 0.25, n_vals).tolist()

    # Precomputed start vector.
    beta = [0.07302293, 0., 0.0254573, -0.04505474, -0.03886735, 0.,
            -0.03806559, 0., 0., 0., 0.01746617, 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.01615937, 0., -0.01359646, 0.02150195, 0.,
            0.02436873, 0., 0., -0.0272448, 0., 0., 0.04213882, 0.03587526,
            0.07456545, 0.06753988, 0.12180282, 0.11120493, 0.15486321,
            0.05453602, 0.16154324, 0.16140251, 0.13619467, 0.20804703,
            0.1832236, 0.19791729, 0.20980641, 0.15999879, 0.19648151,
            0.14527259, 0.23884876, 0.26351719, 0.1643164, 0.29524708,
            0.22982587, 0.26201468, 0.27257074, 0.28497038, 0.2055798,
            0.31811997, 0.22757443, 0.25140015, 0.28259666, 0.40201002]
    beta = np.array(beta).reshape(p, 1)

    err_beta = np.zeros((n_vals, n_vals))
    err_f = np.zeros((n_vals, n_vals))

    # Perform grid search.
    for i in range(len(ls)):
        l = ls[i]
        k = 1.0 - l
        for j in range(len(gs)):
            g = gs[j]

            # Create the loss function.
            function = LinearRegressionL1L2GL(X, y, l, k, g, A=A,
                                              penalty_start=1, mean=False)

            # Create the estimator.
            lr = estimators.LinearRegressionL1L2GL(l1=l, l2=k, gl=g, A=A,
                                     algorithm=CONESTA(max_iter=max_iter,
                                                       eps=eps),
                                     penalty_start=1, mean=False)
            # Fit data with the new regularisation parameters.
            beta = lr.fit(X, y, beta).beta

            # Compute output.
            err_beta[i, j] = np.linalg.norm(beta - beta_star)
            err_f[i, j] = np.linalg.norm(function.f(beta)
                                         - function.f(beta_star))

            print "l: %.3f, g: %.3f, err_f: %.12f" % (l, g, err_f[i, j])

    print "err_beta:\n", err_beta
    print "err_f:\n", err_f

    # Plot the results.
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import pylab

#    np.random.seed(42)
    random_state.set_state(state)

    # Plot results.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(ls, gs)
    Z = err_f

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           antialiased=False, linewidth=0)
    # antialiased=False
    # ax.set_zlim(-1.01, 1.01)
    ax.patch.set_facecolor('none')
    ax.view_init(azim=-30, elev=30)

    plt.xlabel("$\lambda$", fontsize=14)
    plt.ylabel("$\gamma$", fontsize=14)
    plt.title(r"$f(\beta^{(k)}) - f(\beta^*)$", fontsize=16)

    x, y, _ = proj3d.proj_transform(0.618, 1.618, np.min(Z), ax.get_proj())
    label = pylab.annotate(
        "$(0.618, 1.618)$", fontsize=14, xy=(x, y), xytext=(60, 20),
        textcoords='offset points', ha='right', va='bottom', color="white",
        # bbox=dict(boxstyle='round, pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0',
                        color="white"))

    def update_position(e):
        x, y, _ = proj3d.proj_transform(0.618, 1.618, np.min(Z), ax.get_proj())
        label.xy = x, y
        label.update_positions(fig.canvas.renderer)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_release_event', update_position)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf)  # , shrink=0.5, aspect=5)

#    plt.savefig('lr_en_gl.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    lr_en_gl()
