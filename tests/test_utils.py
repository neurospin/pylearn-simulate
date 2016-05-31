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


class TestUtils(TestCase):

    def test_rand(self):

        R = np.array([[-0.25091976, 0.90142861, 0.46398788],
                      [0.19731697, -0.68796272, -0.68801096]])

        rs = np.random.RandomState(42)
        rng = utils.RandomUniform(-1, 1, random_state=rs)
        R2 = rng(2, 3)

        assert(np.linalg.norm(R - R2) < utils.TOLERANCE)

        rng = utils.ConstantValue(31415)
        R = rng(3, 2)

        assert(np.linalg.norm(R - 31415) < utils.TOLERANCE)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()
