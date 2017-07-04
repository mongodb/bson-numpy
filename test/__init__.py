import os
import sys
from functools import wraps

import bson
import bsonnumpy
import numpy as np
import pymongo
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument

if sys.version_info[:2] == (2, 6):
    import unittest2 as unittest
    from unittest2 import SkipTest
else:
    import unittest
    from unittest import SkipTest

PY3 = sys.version_info[0] >= 3

host = bson.py3compat._unicode(os.environ.get("DB_IP", 'localhost'))
port = int(os.environ.get("DB_PORT", 27017))
pair = '%s:%d' % (host, port)


class ClientContext(object):
    """
    ClientContext from PyMongo test suite. May eventually need more _require
    functions, but for now only care if we have a server connection.
    """

    def __init__(self):
        try:
            client = pymongo.MongoClient(host, port,
                                         serverSelectionTimeoutMS=100)
            client.admin.command('ismaster')  # Can we connect?

        except pymongo.errors.ConnectionFailure:
            self.connected = False
            self.client = None
        else:
            self.connected = True
            self.client = pymongo.MongoClient(host, port, connect=False)

    def _require(self, condition, msg, func=None):
        def make_wrapper(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                # Always raise SkipTest if we can't connect to MongoDB
                if not self.connected:
                    raise SkipTest("Cannot connect to MongoDB on %s" % pair)
                if condition:
                    return f(*args, **kwargs)
                raise SkipTest(msg)

            return wrap

        if func is None:
            def decorate(f):
                return make_wrapper(f)

            return decorate
        return make_wrapper(func)

    def require_connected(self, func):
        return self._require(self.connected,
                             "Cannot connect to MongoDB on %s" % pair,
                             func=func)


client_context = ClientContext()

class TestBsonNumpy(unittest.TestCase):
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    def compare_elements(self, expected, actual, dtype):
        if isinstance(expected, dict):
            for key, value in expected.items():
                self.compare_elements(value, actual[key],
                                      dtype=dtype.fields[key][0])

        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))

            # If an array's shape is (3,2), its subarrays' shapes are (2,).
            subdtype, shape = dtype.subdtype
            self.assertGreaterEqual(len(shape), 1)
            subarray_dtype = np.dtype((subdtype, shape[1:]))
            for i in range(len(expected)):
                self.compare_elements(expected[i], actual[i], subarray_dtype)

        elif dtype.kind == 'V':
            self.assertEqual(bytes(expected.ljust(dtype.itemsize, b'\0')),
                             bytes(actual))

        elif dtype.kind == 'S':
            # NumPy only stores bytes, not str.
            self.assertEqual(expected, actual.decode('utf-8'))

        else:
            self.assertEqual(expected, actual)



class TestToNdarray(TestBsonNumpy):
    @classmethod
    def setUpClass(cls):
        cls.client = client_context.client

    def compare_seq_to_ndarray_result(self, np_type, document):
        data = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype(np_type)
        result = bsonnumpy.sequence_to_ndarray([data], dtype, 1)
        self.assertEqual(result.dtype, dtype)
        for key in document:
            self.assertEqual(result[0][key], document[key],
                             "Comparison failed for type %s: %s != %s" % (
                                 dtype, result[0][key], document[key]))


    # TODO: deal with both name and title in dtype
    def compare_results(self, dtype, expected, actual):
        self.assertEqual(dtype, actual.dtype)
        self.assertEqual(expected.count(), len(actual))
        for act in actual:
            exp = next(expected)
            for name in dtype.names:
                self.compare_elements(exp[name], act[name],
                                      dtype=dtype.fields[name][0])

    def get_cursor_sequence(self, docs):
        self.client.bsonnumpy_test.coll.delete_many({})
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        return raw_coll

    def make_mixed_collection_test(self, docs, dtype):
        raw_coll = self.get_cursor_sequence(docs)

        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in raw_coll.find()), dtype, raw_coll.count())
        self.compare_results(np.dtype(dtype),
                             self.client.bsonnumpy_test.coll.find(),
                             ndarray)


class TestToSequence(TestBsonNumpy):
    def compare_results(self, expected, actual, dtype):
        dict_result = [bson._bson_to_dict(
            d, bson.DEFAULT_CODEC_OPTIONS) for d in actual]

        self.assertEqual(len(dict_result), len(expected))
        for i in range(len(dict_result)):
            self.compare_elements(expected[i], dict_result[i], dtype)


def millis(delta):
    if hasattr(delta, 'total_seconds'):
        return delta.total_seconds() * 1000

    # Python 2.6.
    return ((delta.days * 86400 + delta.seconds) * 1000 +
            delta.microseconds / 1000.0)
