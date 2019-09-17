==========
BSON-NumPy
==========
:Info: See `the mongo site <http://www.mongodb.org>`_ for more information. See `GitHub <http://github.com/mongodb/bson-numpy>`_ for the latest source.
:Author: Anna Herlihy
:Maintainer: The MongoDB Python Team
:Documentation: https://bson-numpy.readthedocs.io/en/latest/

About
=====

A Python extension written in C that uses `libbson
<http://mongoc.org/libbson/current>`_ to convert between `NumPy <https://pypi.org/project/numpy/>`_
arrays and `BSON <http://bsonspec.org>`_, the native data format of MongoDB.

This project is a **prototype** and not ready for production use.

Bugs / Feature Requests
=======================

Think you’ve found a bug? Want to see a new feature in BSON-NumPy? `Please open a
issue in github. <https://github.com/mongodb/bson-numpy/issues>`_

How To Ask For Help
-------------------

Please include all of the following information when opening an issue:

- Detailed steps to reproduce the problem, including full traceback, if possible.
- The exact python version used, with patch level::

  $ python -c "import sys; print(sys.version)"

- The operating system and version (e.g. Windows 7, OSX 10.8, ...)

Installation
============

Please see the `instructions on readthedocs.
<https://bson-numpy.readthedocs.io/en/latest/#installing>`_

Dependencies
============

BSON-NumPy supports CPython 3.5+. BSON-NumPy requires NumPy 1.17.0+.

Examples
========

Please see the `examples on readthedocs.
<https://bson-numpy.readthedocs.io/en/latest/#converting-mongodb-data-to-numpy>`_

Documentation
=============

Please see the `documentation on readthedocs.
<https://bson-numpy.readthedocs.io/en/latest/>`_

Testing
=======

The easiest way to run the tests is to run **python setup.py test** in
the root of the distribution.

.. _sphinx: http://sphinx.pocoo.org/
