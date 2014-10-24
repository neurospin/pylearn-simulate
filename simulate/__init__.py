# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:26:14 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import beta
import correlation_matrices
import functions
import utils

from .simulate import LinearRegressionData

__all__ = ["LinearRegressionData",
           'beta', 'correlation_matrices', 'functions',
           'utils']
