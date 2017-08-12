BSON-NumPy: Fast Conversion Library
===================================

A Python extension written in C that uses `libbson
<http://mongoc.org/libbson/current>`_ to convert between NumPy arrays and BSON,
the native data format of MongoDB.

This is currently a **prototype**. It requires PyMongo installed from a branch.
See the `installing`_ instructions below.

Converting MongoDB data to NumPy
--------------------------------

.. testsetup::

    import bson
    import bsonnumpy
    import numpy as np
    import pymongo

    c = pymongo.MongoClient()
    c.test.collection.delete_many({})
    c.test.collection.insert_many([
        {'_id': 1, 'n': 1.0, 'str': 'hello'},
        {'_id': 2, 'n': 3.1, 'str': 'and'},
        {'_id': 3, 'n': 7.7, 'str': 'goodbye'},
    ])

Say we have a collection in MongoDB with three documents::

    {'_id': 1, 'n': 1.5, 'str': 'hello'}
    {'_id': 2, 'n': 3.1, 'str': 'and'}
    {'_id': 3, 'n': 7.7, 'str': 'goodbye'}

We can convert these to a NumPy :class:`~numpy.ndarray` directly:

.. Comment: duplicate some testsetup imports here for readers to see.
   We still need them in testsetup, however, so that we don't have to repeat
   them in the doctest blocks below.

.. doctest::

    >>> from pymongo import MongoClient
    >>> import numpy as np
    >>> import bsonnumpy
    >>>
    >>> client = MongoClient()
    >>> collection = client.test.collection
    >>> dtype = np.dtype([('_id', np.int64), ('n', np.double), ('str', 'S10')])
    >>> ndarray = bsonnumpy.sequence_to_ndarray(
    ...     collection.find_raw(), dtype, collection.count())
    >>>
    >>> print(ndarray)
    [(1,  1. , b'hello') (2,  3.1, b'and') (3,  7.7, b'goodbye')]
    >>> print(ndarray.dtype)
    [('_id', '<i8'), ('n', '<f8'), ('str', 'S10')]

Using MongoDB with Pandas
-------------------------

The :class:`~numpy.ndarray` created above can be wrapped in a Pandas DataFrame:

.. doctest::

    >>> import pandas as pd
    >>> pd.DataFrame(ndarray, index=ndarray['_id'])
       _id    n         str
    1    1  1.0    b'hello'
    2    2  3.1      b'and'
    3    3  7.7  b'goodbye'

API
---

.. py:function:: sequence_to_ndarray(iterator, dtype, length)

  Convert a series of bytes objects, each containing raw BSON data, into a
  NumPy array.

  Parameters:

  - `iterator`: A :ref:`sequence <typesseq>` or :ref:`iterator <typeiter>` representing a sequence
    of :class:`bytes` objects containing BSON documents.
  - `dtype`: A :class:`numpy.dtype` listing the fields to extract from each
    BSON document and what NumPy type to convert it to.
  - `length`: An integer, the number of items in `iterator`.

  Returns an :class:`~numpy.ndarray`. If the length of `iterator` is not the same
  as the `length` argument to :func:`sequence_to_ndarray`, the returned array's
  length is the shorter of the two.

.. py:exception:: bsonnumpy.error

  Raised by any runtime error in the module.

Installing
----------

BSON-NumPy is supported on Linux and Mac OS X, with Python 2.6 and later,
on Intel architectures. It requires NumPy 1.11 or greater.

The current prototype of BSON-NumPy relies on PyMongo installed from a branch::

  $ python3 -m pip install numpy
  $ python3 -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python3 -m pip install git+https://github.com/mongodb/bson-numpy.git

Or in Python 2::

  $ python -m pip install numpy
  $ python -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python -m pip install git+https://github.com/mongodb/bson-numpy.git

Here are more detailed instructions for a few platforms.

Debian or Ubuntu
^^^^^^^^^^^^^^^^

Install with Python 3::

  $ sudo apt-get install -y python3-dev python3-numpy python3-pip
  $ python3 -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python3 -m pip install git+https://github.com/mongodb/bson-numpy.git

Or Python 2::

  $ sudo apt-get install -y python-dev python-numpy python-pip
  $ python -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python -m pip install git+https://github.com/mongodb/bson-numpy.git

Fedora or RedHat
^^^^^^^^^^^^^^^^

With Python3::

  $ sudo yum install -y python3-devel python3-numpy python3-pip
  $ python3 -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python3 -m pip install git+https://github.com/mongodb/bson-numpy.git

Or Python 2::

  $ sudo yum install -y python-devel numpy python-pip
  $ python -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python -m pip install git+https://github.com/mongodb/bson-numpy.git

Mac OS X
^^^^^^^^

The easiest way to install BSON-NumPy's dependencies is with Homebrew.

* `Install Homebrew <http://brew.sh/>`_

macOS comes with an outdated version of NumPy, too old to work with BSON-NumPy.
We recommend you don't use the macOS system Python at all, and install your own
Python with ``brew install python3`` or ``brew install python``, or `download
Python from python.org <https://www.python.org/downloads/>`_. Then::

  $ python3 -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python3 -m pip install git+https://github.com/mongodb/bson-numpy.git

Or in Python 2::

  $ python -m pip install -U git+https://github.com/ajdavis/mongo-python-driver.git@raw-batch
  $ python -m pip install git+https://github.com/mongodb/bson-numpy.git

Converting BSON to NumPy
------------------------

The following examples use Python 3.6 and NumPy 1.12.

Double, int32, int64
^^^^^^^^^^^^^^^^^^^^

BSON numeric types convert naturally:

.. doctest::

    >>> data = bson.BSON().encode({'pi': 3.14159, 'answer': 42, 'big': 2**63-1})
    >>> dtype = np.dtype([('pi', np.double), ('answer', np.int32), ('big', np.int64)])
    >>> bsonnumpy.sequence_to_ndarray([data], dtype, 1)
    array([( 3.14159, 42, 9223372036854775807)],
          dtype=[('pi', '<f8'), ('answer', '<i4'), ('big', '<i8')])

Arrays
^^^^^^

An embedded array in BSON becomes an additional dimension in NumPy:

.. doctest::

    >>> data = bson.BSON().encode({'a': [1, 2, 3]})
    >>> bsonnumpy.sequence_to_ndarray([data],
    ...                               np.dtype([('a', '3i')]),
    ...                               1)
    array([([1, 2, 3],)],
          dtype=[('a', '<i4', (3,))])

Nested documents
^^^^^^^^^^^^^^^^

Access fields of nested BSON documents by declaring a nested dtype:

.. doctest::

    >>> data = bson.BSON().encode({'a': {'b': 1, 'c': 3.14}})
    >>> dtype = np.dtype([('a',
    ...                    np.dtype([('b', 'i'), ('c', 'f8')]))])
    >>> array = bsonnumpy.sequence_to_ndarray([data], dtype, 1)
    >>> array
    array([((1,  3.14),)],
          dtype=[('a', [('b', '<i4'), ('c', '<f8')])])

The values can be retrieved by name or by position:

.. doctest::

    >>> array[0]
    ((1,  3.14),)
    >>> array[0]['a']
    (1,  3.14)
    >>> array[0]['a']['b']
    1
    >>> array[0]['a']['c']
    3.1400000000000001
    >>> array[0][0][1]
    3.1400000000000001

Binary
^^^^^^

Convert BSON binary data to NumPy with type "V" (void) or "S" (string), and a
fixed length:

.. doctest::

    >>> doc1 = bson.BSON().encode({'a': bson.Binary(b'binary data')})
    >>> doc2 = bson.BSON().encode({'a': bson.Binary(b'short')})
    >>> array = bsonnumpy.sequence_to_ndarray([doc1, doc2],
    ...                                       np.dtype([('a', 'V10')]),
    ...                                       2)
    >>> array[0][0].tobytes()
    b'binary dat'
    >>> array[1][0].tobytes()
    b'short\x00\x00\x00\x00\x00'

This example uses the format "V10" for 10 bytes of untyped data. Notice that
BSON-NumPy truncates the longer byte string to 10 bytes, and zero-pads the
shorter one.

Strings
^^^^^^^

Convert BSON UTF-8 strings the same as binary, with type "V" or "S" and a
fixed length. As with binary data, BSON-NumPy truncates or zero-extends the
input data to match the dtype length:

.. doctest::

    >>> data = bson.BSON().encode({'x': 'to be or not to be'})
    >>> bsonnumpy.sequence_to_ndarray([data], np.dtype([('x', 'S5')]), 1)
    array([(b'to be',)],
          dtype=[('x', 'S5')])

Bool
^^^^

Convert BSON bools to NumPy bools with the "b" specifier:

.. doctest::

    >>> data = bson.BSON().encode({'x': True, 'y': False})
    >>> bsonnumpy.sequence_to_ndarray([data],
    ...                               np.dtype([('x', 'b'), ('y', 'b')]),
    ...                               1)
    array([(1, 0)],
          dtype=[('x', 'i1'), ('y', 'i1')])

Datetime
^^^^^^^^

BSON datetimes become 64-bit Unix timestamps (milliseconds since January 1,
1970 UTC):

.. doctest::

    >>> from datetime import datetime
    >>> data = bson.BSON().encode({'when': datetime(2017, 1, 1)})
    >>> bsonnumpy.sequence_to_ndarray([data],
    ...                               np.dtype([('when', np.int64)]),
    ...                               1)
    array([(1483228800000,)],
          dtype=[('when', '<i8')])

ObjectId
^^^^^^^^

ObjectIds are 12 bytes long. Use "V12" or "S12" to convert ObjectIds to untyped
data or byte strings:

.. doctest::

    >>> oid = bson.ObjectId('588a6aefa08bff08f62a66c7')
    >>> data = bson.BSON().encode({'_id': oid})
    >>> bsonnumpy.sequence_to_ndarray([data], np.dtype([('_id', 'S12')]), 1)
    array([(b'X\x8aj\xef\xa0\x8b\xff\x08\xf6*f\xc7',)],
          dtype=[('_id', 'S12')])

Not supported
^^^^^^^^^^^^^

`File an issue <https://github.com/mongodb/bson-numpy/issues>`_
if you need support for any of the following BSON types.

* Code
* Code with scope
* DBPointer
* Decimal 128
* Min Key
* Max Key
* Null
* Regular Expression
* Symbol
* Timestamp
* Undefined
