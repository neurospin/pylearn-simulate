# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:16:19 2015

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

params = dict(name="pylearn-simulate",
              version="0.2",
              author="See contributors on https://github.com/neurospin/pylearn-simulate",
              author_email="lofstedt.tommy@gmail.com",
              maintainer="Tommy Löfstedt",
              maintainer_email="lofstedt.tommy@gmail.com",
              description="pylearn-simulate: Generate theoretically sound simulated data",
              long_description=read('README.md'),
              license="BSD 3-clause.",
              keywords="simulated data, simulation, machine learning, structured, sparse, regularization, penalties",
              url="https://github.com/neurospin/pylearn-simulate",
              package_dir={"": "./"},
              packages=["simulate",
                       ],
#              package_data = {"": ["README.md", "LICENSE"],
#                              "examples": ["*.py"],
#                             },
              classifiers=["Development Status :: 3 - Alpha",
                           "Intended Audience :: Developers",
                           "Intended Audience :: Science/Research",
                           "License :: OSI Approved :: BSD 3-Clause License",
                           "Topic :: Scientific/Engineering",
                           "Programming Language :: Python :: 2.7",
                          ],
)

try:
    from setuptools import setup

    params["install_requires"] = ["numpy>=1.6.1",
                                  "scipy>=0.9.0",
                                 ]
    params["extras_require"] = {"examples": ["parsimony>=0.2.1", "matplotlib>=1.1.1rc"],
                                "test": ["doctest"],
                               }
except:
    from distutils.code import setup

setup(**params)
