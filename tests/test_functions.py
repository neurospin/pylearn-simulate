# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 14:29:01 2016

Copyright (c) 2013-2016, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest

import numpy as np

from tests import TestCase
import simulate.beta as beta
import simulate.utils as utils
import simulate.functions as functions


class TestFunctions(TestCase):

    def test_functions(self):

        rs = np.random.RandomState(1337)
        rng01 = utils.RandomUniform(0, 1, random_state=rs)
        rng11 = utils.RandomUniform(-1, 1, random_state=rs)
        cnst0 = utils.ConstantValue(0, random_state=rs)

        x = beta.random((10, 1), density=0.75, rng=rng11, sort=False,
                        normalise=False)

        # L1
        grad = np.array([[0.57911668],
                         [-1.0],
                         [1.0],
                         [-1.0],
                         [-1.0],
                         [-1.0],
                         [1.0],
                         [-1.0],
                         [0.58823715],
                         [-1.0]])

        assert(np.linalg.norm(0.1 * grad - functions.L1(0.1, rng=rng11).grad(x)) < utils.TOLERANCE)

        grad = np.array([[0.0],
                         [-0.7159978],
                         [0.0735713],
                         [-0.1627325],
                         [-0.9519013],
                         [-1.0],
                         [1.0],
                         [-0.8874939],
                         [0.0],
                         [-0.9522283]])

        assert(np.linalg.norm(0.1 * grad - functions.SmoothedL1(0.1, mu=5e-1).grad(x)) < utils.TOLERANCE)

        grad = np.array([[0.0],
                         [-1.0],
                         [1.0],
                         [-1.0],
                         [-1.0],
                         [-1.0],
                         [1.0],
                         [-1.0],
                         [0.0],
                         [-1.0]])

        assert(np.linalg.norm(0.25 * grad - functions.SmoothedL1(0.25, mu=5e-8).grad(x)) < utils.TOLERANCE)

        x = beta.random((10, 1), density=0.5, rng=rng11, sort=False,
                        normalise=False)

        # L2
        grad = np.array([[0.0],
                         [-0.31136769],
                         [-0.18828614],
                         [0.58389796],
                         [0.0],
                         [0.0],
                         [-0.70064494],
                         [0.0],
                         [0.0],
                         [0.18909872]])

        assert(np.linalg.norm(0.1 * grad - functions.L2(0.1, rng=rng01).grad(x)) < utils.TOLERANCE)
        assert(abs(np.linalg.norm(functions.L2(0.5, rng=rng01).grad(x)) < 0.5) < utils.TOLERANCE)
        assert(abs(np.linalg.norm(functions.L2(1.0, rng=rng01).grad(x)) < 1.0) < utils.TOLERANCE)
        assert(abs(np.linalg.norm(functions.L2(1.5, rng=rng01).grad(x)) < 1.5) < utils.TOLERANCE)

        # L2 Squared
        assert(np.linalg.norm(0.5 * x - functions.L2Squared(0.5).grad(x)) < utils.TOLERANCE)

        # Total Variation
        x = beta.random((10, 10), density=0.5, rng=rng11, sort=True,
                        normalise=False)

        # Generate the linear operator for total variation.
        A = functions.TotalVariation.A_from_shape((10, 10))
        A_ = np.array([[-1.0, 1.0, 0.0],
                       [0.0, -1.0, 1.0],
                       [0.0, 0.0, -1.0]])
        assert(np.linalg.norm(A[0].todense()[:3, :3] - A_) < utils.TOLERANCE)
        A_ = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(np.linalg.norm(A[1].todense()[:10, 5:15] - A_) < utils.TOLERANCE)

        grad = np.array([[-1.37690875], [-0.50249731], [-1.13463951],
                         [-0.92581747], [-1.12545778], [-0.72495808],
                         [-1.09539063], [-1.12191557], [-1.14278035],
                         [-0.38378842], [-0.54044045], [0.1998922],
                         [0.10727523], [0.42029647], [-0.12845922],
                         [-0.09340557], [0.0055352], [0.0456007],
                         [0.8779322], [1.40188876], [-0.57962065],
                         [0.35934782], [-0.42347626], [0.29359706],
                         [2.07205284], [0.65534416], [1.01918068],
                         [-0.45464203], [0.38998718], [-0.19745865],
                         [-0.62748016], [0.63761495], [-0.68385703],
                         [1.89910956], [-0.28092104], [-0.17162572],
                         [0.48321381], [0.79706197], [0.61187108],
                         [-1.42531316], [0.7902697], [0.60837155],
                         [1.20533504], [0.09427085], [-0.70922127],
                         [0.09077005], [0.14231686], [-0.32261771],
                         [-1.0636391], [0.3522861], [0.46381846],
                         [-0.32764258], [-0.9267115], [0.68692098],
                         [-0.28524282], [-1.65222339], [0.3472565],
                         [-0.82272974], [0.60190561], [0.43221722],
                         [0.92882439], [0.21409408], [-0.29173676],
                         [0.43373039], [-0.52463849], [1.59982977],
                         [-1.20161831], [-1.40845266], [0.34302998],
                         [0.90820945], [0.79001083], [0.06567102],
                         [-0.31555666], [-1.34826493], [-1.31252716],
                         [-1.1824102], [-0.31668549], [-0.1015793],
                         [-0.19644832], [0.97835737], [-1.23182642],
                         [-0.63448914], [-1.29006382], [-0.00488059],
                         [-0.34004922], [0.4112866], [-0.36592392],
                         [0.06443214], [-1.17634974], [0.58974244],
                         [-0.55508336], [1.0], [0.99971041],
                         [0.99956841], [0.86125816], [0.99525803],
                         [0.96641958], [0.99990113], [0.80759139], [2.0]])

        assert(np.linalg.norm(functions.TotalVariation(0.1, A, rng=rng01).grad(x) - 0.1 * grad) < utils.TOLERANCE)
        tvgrad = functions.TotalVariation(0.1, A, rng=cnst0).grad(x)
        tvmugrad = functions.SmoothedTotalVariation(0.1, A, mu=5e-8).grad(x)
        assert(np.linalg.norm(tvgrad - tvmugrad) < utils.TOLERANCE)

        # Group Lasso
        x = beta.random((10, 1), density=0.5, rng=rng11, sort=False,
                        normalise=False)
        # Create linear operator.
        A = functions.GroupLasso.A_from_groups(10, [range(5), range(5, 10)])
        glgrad = functions.GroupLasso(1.0, A, rng=rng11).grad(x)
        assert(abs(np.linalg.norm(glgrad[:5]) - 1.0) < utils.TOLERANCE)
        assert(abs(np.linalg.norm(glgrad[5:]) - 1.0) < utils.TOLERANCE)
        glmugrad = functions.SmoothedGroupLasso(1.0, A, mu=5e-8).grad(x)
        assert(np.linalg.norm(glgrad - glmugrad) < utils.TOLERANCE)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()
