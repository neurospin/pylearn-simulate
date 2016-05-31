# -*- coding: utf-8 -*-
"""
The :mod:`simulate.tests` package includes tests for all (in time, at least)
modules and functions.

Copyright (c) 2013-2016, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
from .tests import TestCase
from .tests import test_all

__all__ = ["TestCase", "test_all"]
