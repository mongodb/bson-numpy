BSON-NumPy: Fast Conversion Library
===================================

A Python extension written in C that uses `libbson
<http://mongoc.org/libbson/current>`_ to convert between NumPy arrays and BSON,
the native data format of MongoDB.

Converting MongoDB data to NumPy
--------------------------------

.. testsetup::

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

.. doctest::

  >>> from bson import CodecOptions
  >>> from bson.raw_bson import RawBSONDocument
  >>> from pymongo import MongoClient
  >>> import numpy as np
  >>> import bsonnumpy
  >>>
  >>> client = MongoClient()
  >>> collection = client.test.get_collection(
  ...     'collection',
  ...     codec_options=CodecOptions(document_class=RawBSONDocument))
  >>>
  >>> dtype = np.dtype([('_id', np.int64), ('n', np.double), ('str', 'S10')])
  >>> ndarray = bsonnumpy.sequence_to_ndarray(
  ...    (doc.raw for doc in collection.find()), dtype, collection.count())
  >>>
  >>> print(ndarray)
  [(1, 1.0, 'hello') (2, 3.1, 'and') (3, 7.7, 'goodbye')]
  >>> print(ndarray.dtype)
  [('_id', '<i8'), ('n', '<f8'), ('str', 'S10')]

API
---

.. py:function:: sequence_to_ndarray(iterator, dtype, length)

  Convert a series of bytes objects, each containing raw BSON data, into a
  NumPy array.

  Parameters:

  - `iterator`: An :ref:`iterator object <typeiter>` representing a sequence
    of :class:`bytes` objects containing BSON documents.
  - `dtype`: A :class:`numpy.dtype` listing the fields to extract from each
    BSON document and what NumPy type to convert it to.
  - `length`: An integer, the number of items in `iterator`.

Installing
----------

BSON-NumPy is supported on Linux and Mac OS X, with Python 2.6 and later.
Installation is in two steps: first install the C library libbson yourself,
then install BSON-NumPy with pip:

- `Install libbson <http://mongoc.org/libbson/current/installing.html>`_
- ``python -m pip install bsonnumpy``
