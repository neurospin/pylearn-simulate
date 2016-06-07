pylearn-simulate
================

Simulated data are widely used to assess optimisation methods. This is because
of their ability to evaluate certain aspects of the methods under study, that
are impossible to look into when using real data sets. In the context of convex
optimisation, it is never possible to know the exact solution of the
minimisation problem with real data and it is a difficult problem even with
simulated data. We propose to generalise an approach originally published by
Nesterov (2013), for LASSO regression, to a broader family of penalised
regression problems.

We would like to generate simulated data for which we know the exact solution
of the optimised function. The inputs are: The minimiser `b*` (p-by-1), a
candidate data set `X0` (n-by-p), residual vector `e` (n-by-1), regularisation
parameters (in our case they are two: `k` and `g`), the signal-to-noise ratio
`s`, and the expression of the function `f(b)` to minimise.

The candidate version of the dataset may for instance be `X0 ~ N(m, S)`, and
the residual vector may be `e ~ N(0, 1)`.

The proposed procedure outputs `X` and `y` such that
```
    b* = argmin f(b, X0, e, k, g, s),
```
with `f` a convex function that depends on the parameters defining the
simulated data.

Oftentimes in linear regression, simulated data are generated such that
```
  (1)    y = X * b* + e.
```
If we want to evaluate an algorithm to minimise the LASSO problem
```
  (2)    0.5 * ||X * b - y||² + l * |b|,
```
where `||.||²` is the squared L2-norm and `|.|` is the L1-norm, then we need to
use e.g. cross-validation to find `l`. But the found `l` is very likely
suboptimal, and in any case, we are forced to compare the solution to (1),
which is not sparse.

This package thus provides the solution that minimises (2), instead of
(1), namely `b*` and `l`. Which means that you will be able to compare both
speed, sensitivity to noise, correlation, etc., and the actual solutions of
different minimisation algorithms.

With this package, `pylearn-simulate`, it is straight-forward to generate such
data. `pylearn-simulate` is written for Python 2.7.x.





Dependencies
------------
The reference environment for pylearn-simulate is Ubuntu 12.04 LTS with
Python 2.7.3, Numpy 1.6.1 and Scipy 0.9.0. More recent versions likely work,
but have not been tested thoroughly.

Unless you already have Numpy and Scipy installed, you need to install them:
```
$ sudo apt-get install python-numpy python-scipy
```

In order to show plots and to run the examples, you may need to install
Matplotlib:
```
$ sudo apt-get install python-matplotlib
```

Unless you have your own minimising software, we recommend that you download
and install [pylearn-parsimony](https://github.com/neurospin/pylearn-parsimony)
([https://github.com/neurospin/pylearn-parsimony](https://github.com/neurospin/pylearn-parsimony)).
While pylearn-parsimony is not a requirement to use pylearn-simulate, the
examples are using it. Thus, in order to to run all parts of the examples, you
will need pylearn-parsimony.





Installation
------------
### Easiest installation
Download `pylearn-simulate` and put it in your `PYTHONPATH`. Alternatively, put
the directory `pylearn-simulate/simulate/` in the directory of your project.

### Recommended installation
Download the release of pylearn-simulate from
[https://github.com/neurospin/pylearn-simulate/releases](https://github.com/neurospin/pylearn-simulate/releases).
Unpack the file, go to the pylearn-simulate directory and type:
```
$ python setup.py install --user
```
for a local installation in the user's *userbase* directory (usually in
`~/.local/lib/python2.7/site-packages` on Unix-like/-based operating systems,
such as Linux and OS X, and in
`%AppData%\Python\Python27\site-packages` on Windows), or
```
$ sudo python setup.py install
```
for a global installation accessible to all users. You will need to have administrator rights on
your computer in order to install software for all users.

You are now ready to use your fresh installation of pylearn-simulate!





Examples
--------
See the examples in `pylearn-simulate/examples/`.
