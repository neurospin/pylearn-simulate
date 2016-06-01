# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 14:01:01 2016

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


class TestBeta(TestCase):

    def test_beta(self):

        rs = np.random.RandomState(42)
        rng01 = utils.RandomUniform(0, 1, random_state=rs)

        x = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.15601864],
                      [0.37454012],
                      [0.59865848],
                      [0.73199394],
                      [0.95071431]])

        x2 = beta.random((10, 1), density=0.5, rng=rng01, sort=True,
                         normalise=False)

        assert(np.linalg.norm(x - x2) < utils.TOLERANCE)

        rs = np.random.RandomState(1337)
        rng11 = utils.RandomUniform(-1, 1, random_state=rs)

        x = np.array([[0.0],
                      [-0.24365074],
                      [0.02503597],
                      [-0.0553771],
                      [-0.3239276],
                      [-0.46459304],
                      [0.64803846],
                      [-0.30201006],
                      [0.0],
                      [-0.32403887]])

        x2 = beta.random((10, 1), density=0.75, rng=rng11, sort=False,
                         normalise=True)

        assert(np.linalg.norm(x - x2) < utils.TOLERANCE)
        assert(abs(np.linalg.norm(x2) - 1) < utils.TOLERANCE)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()
