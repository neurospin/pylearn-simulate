# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:50:17 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import numpy as np

__all__ = ['TOLERANCE', 'RandomUniform', 'ConstantValue',
           'find_bisect_interval']

TOLERANCE = 5e-8


class RandomNumberGenerator(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, random_state=None):

        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state

    @abc.abstractmethod
    def __call__(self, *d):
        raise NotImplementedError('Abstract method "__call__" must be '
                                  'specialised!')

    def set_state(self, state):
        self.random_state.set_state(state)

    def get_state(self):
        return self.random_state.get_state()


class RandomUniform(RandomNumberGenerator):
    """Random number generator that returns a uniformly distributed value.

    Example
    -------
    >>> rnd = RandomUniform(-1, 1)
    >>> rnd(3) #doctest: +ELLIPSIS
    array([...])
    >>> rnd(2, 2) #doctest: +ELLIPSIS
    array([[..., ...],
           [..., ...]])
    """
    def __init__(self, a=0.0, b=1.0, random_state=None):

        super(RandomUniform, self).__init__(random_state=random_state)

        a = float(a)
        b = float(b)
        a, b = min(a, b), max(a, b)
        self.a = a
        self.b = b

    def __call__(self, *d):

        if self.random_state is not None:
            R = self.random_state.uniform(self.a, self.b, d)
        else:
            R = np.random.rand(*d)
            R = R * (self.b - self.a) + self.a

        return R


class ConstantValue(RandomNumberGenerator):
    """Random-like number generator that returns a constant value.

    Example
    -------
    >>> rnd = ConstantValue(5.)
    >>> rnd(3)
    array([ 5.,  5.,  5.])
    >>> rnd(2, 2)
    array([[ 5.,  5.],
           [ 5.,  5.]])
    """
    def __init__(self, val, random_state=None):

        super(ConstantValue, self).__init__(random_state=random_state)

        self.val = val

    def __call__(self, *shape):

        return np.repeat(self.val, np.prod(shape)).reshape(shape)


def find_bisect_interval(f, low=-1.0, high=1.0, maxiter=100):
    """Finds values of low and high, such that sign(f(low)) != sign(f(high).

    These values can be used in e.g. the Bisection method to find a root of f.

    Parameters
    ----------
    f : function object or callable.
        The function for which a root is to be found.

    low : float
        An initial guess for the lower end of the interval. Default is -1.

    high : float
        An initial guess for the upper end of the interval. Default is 1.

    maxiter : int
        The maximum number of iterations. Default is 100.

    Returns
    -------
    low : float
        The lower end of the interval.

    high : float
        The upper end of the interval.
    """
    low = float(low)
    high = float(high)
    low, high = min(low, high), max(low, high)

    ilow = abs(low)
    ihigh = abs(high)
    for i in range(maxiter):
        l = f(low)
        h = f(high)
        if np.sign(l) != np.sign(h):
            break
        else:
            if low < 0.0:
                low = 2 * low
            else:
                low -= ilow * 1.5 ** i
            if high < 0.0:
                high += ihigh * 1.5 ** i
            else:
                high = 2 * high

    return low, high


if __name__ == "__main__":
    import doctest
    doctest.testmod()
