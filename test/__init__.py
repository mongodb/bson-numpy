import os
import sys
from functools import wraps

import bson
import bsonnumpy
import numpy as np
import pymongo

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


class TestFromBSON(unittest.TestCase):
    def compare_results(self, np_type, document, compare_to):
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype(np_type)
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(result.dtype, dtype)
        for i in range(len(result)):
            self.assertEqual(compare_to[str(i)], result[i],
                             "Comparison failed for type %s: %s != %s" % (
                                 dtype, compare_to[str(i)], result[i]))
