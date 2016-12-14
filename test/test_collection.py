import random
import string

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from bson.py3compat import b
from bson import BSON
import numpy as np

import bsonnumpy
from test import unittest, client_context


class TestCollection2Ndarray(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = client_context.client

    def compare_elements(self, expected, actual):
        # print("comparing:", type(actual))
        # print(actual)
        # print(expected)
        if isinstance(actual, np.ndarray):
            self.assertTrue(isinstance(expected, list) or isinstance(expected, np.ndarray))
            self.assertEqual(len(actual), len(expected))
            for i in range(len(actual)):
                self.compare_elements(expected[i], actual[i])
            pass

        elif isinstance(actual, np.bytes_):
            expected = b(expected)
            self.assertEqual(expected, actual)
        elif isinstance(actual, np.void):
            doc = BSON(actual).decode()
            self.assertEqual(doc, expected)
        else:
            self.assertEqual(expected, actual)


    # TODO: deal with both name and title in dtype
    def compare_results(self, dtype, expected, actual):
        self.assertEqual(dtype, actual.dtype)
        self.assertEqual(expected.count(), len(actual))
        for act in actual:
            exp = next(expected)
            for desc in dtype.descr:
                name = desc[0]
                self.compare_elements(exp[name], act[name])

    def make_mixed_collection_test(self, docs, dtype):
        self.client.drop_database("bsonnumpy_test")
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        ndarray = bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        print "ndarray", ndarray
        print "docs", [tuple(v[k] for k in v.keys() if k != '_id') for v in docs]
        self.compare_results(dtype, self.client.bsonnumpy_test.coll.find(), ndarray)

    @client_context.require_connected
    def test_collection_flexible_int32(self):
        docs = [{"x": i, "y": 10-i} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int32), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_mixed_scalar(self):
        docs = [{"x": i, "y": random.choice(string.ascii_lowercase)*11} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S11'), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray1(self):
        # 2d subarray
        docs = [{"x": [1+i, -i-1], "y": [i, -i]} for i in range(5)]
        dtype = np.dtype([('x', '2int32'), ('y', '2int32')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', '2int32'), ('x', '2int32')])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray2(self):
        # 3d subarray
        docs = [{"x": [[i, i+1, i+2],
                       [-i, -i-1, -i-2],
                       [100*i, 100*i+1, 100*i+2],
                       [0, 1, 2]],
                 "y": "string!!!"} for i in range(5)]
        dtype = np.dtype([('x', "(4,3)int32"), ('y', 'S10')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S10'), ('x', "(4,3)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray3(self):
        # 3d subarray
        docs = [
            {"x": [[[i, i+1], [i+1, i+2], [i+2, i+3]],
                       [[-i, -i+1], [-i-1, -i], [-i-2, -i-1]],
                       [[100*i, 100*i+i], [100*i+1, 100*i+i], [100*i+2, 100*i+i]],
                       [[0,1], [1,2], [3,4]]],
             "some_other_key": ["string" + str(i), "string" + str(i+1)]} for i in range(5)]
        dtype = np.dtype([('x', "(4,3,2)int32"), ('some_other_key', '2S10')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('some_other_key', '2S10'), ('x', "(4,3,2)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray2_mixed1(self):
        # 3d subarray
        docs = [{"x": [[i, i+1, i+2],
                       [-i, -i-1, -i-2],
                       [100*i, 100*i+1, 100*i+2]],
                 "y": 100-i} for i in range(2)]
        dtype = np.dtype([('x', "(3,3)int32"), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int32), ('x', "(3,3)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_mixed(self):
        docs = [{"x": [i, -i], "y": random.choice(string.ascii_lowercase)*11, "z": {"a": i}} for i in range(10)]
        dtype = np.dtype([('x', '2int32'), ('y', 'S11'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype);
        dtype = np.dtype([('z', 'V12'), ('x', '2int32'), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype);
        dtype = np.dtype([('y', 'S11'), ('x', '2int32'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype);

    @client_context.require_connected
    def test_collection_sub1(self):
        docs = [{'x': {'y': 100+i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32)])
        dtype_sub = np.dtype([('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_not_flexible(self):
        # TODO: determine what to do when user doesn't give a flexible type for documents. Doc order?
        docs = [{"x": [i, i-1], "y": [10-i, 9-i]} for i in range(10)]
        dtype = np.dtype("(2, 2)int32")
        # self.make_mixed_collection_test(docs, dtype)
        docs = [{"x": i, "y": 10-i} for i in range(10)]
        dtype = np.dtype("2int32")
        # self.make_mixed_collection_test(docs, dtype)
