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
import simulate.correlation_matrices as corrmat
import simulate.utils as utils


class TestCovMat(TestCase):

    def test_constant(self):

        S1 = corrmat.constant_correlation(p=[2, 3], rho=[0.1, 0.5], delta=0.0,
                                          eps=0.0)

        S2 = np.array([[1.0, 0.1, 0.0, 0.0, 0.0],
                       [0.1, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.5, 0.5],
                       [0.0, 0.0, 0.5, 1.0, 0.5],
                       [0.0, 0.0, 0.5, 0.5, 1.0]])

        assert(np.linalg.norm(S1 - S2) < utils.TOLERANCE)

    def test_toeplitz(self):
        pk = 5
        rhok = 0.5
        v = [0] * (pk - 1)
        for i in xrange(0, pk - 1):
            v[i] = rhok ** (i + 1)

        Sk = np.eye(pk, pk)
        Sk[0, 1:] = v
        Sk[1:, 0] = v
        for i in xrange(1, pk - 1):
            Sk[i, i + 1:] = v[:-i]
            Sk[i + 1:, i] = v[:-i]

        Sk2 = corrmat.toeplitz_correlation(p=pk, rho=rhok, eps=0.0)

        assert(np.linalg.norm(Sk - Sk2) < utils.TOLERANCE)

        Sk = np.array([[1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.25, 0.0625, 0.015625, 0.00390625],
                      [0.0, 0.0, 0.0, 0.25, 1.0, 0.25, 0.0625, 0.015625],
                      [0.0, 0.0, 0.0, 0.0625, 0.25, 1.0, 0.25, 0.0625],
                      [0.0, 0.0, 0.0, 0.015625, 0.0625, 0.25, 1.0, 0.25],
                      [0.0, 0.0, 0.0, 0.00390625, 0.015625, 0.0625, 0.25, 1.0]])

        Sk2 = corrmat.toeplitz_correlation(p=[3, 5], rho=[0.5, 0.25], eps=0.0)

        assert(np.linalg.norm(Sk - Sk2) < utils.TOLERANCE)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()
