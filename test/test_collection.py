import random
import string

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from bson.py3compat import b
import numpy as np

import bsonnumpy
from test import unittest, client_context


class TestCollection2Ndarray(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = client_context.client

    # TODO: deal with both name and title in dtype
    def compare_results(self, dtype, expected, actual):
        self.assertEqual(dtype, actual.dtype)
        self.assertEqual(expected.count(), len(actual))
        for act in actual:
            exp = next(expected)
            for desc in dtype.descr:
                # won't compare complex types
                name = desc[0]
                if isinstance(act[name], np.bytes_):
                    exp[name] = b(exp[name])
                self.assertEqual(exp[name], act[name])


    @client_context.require_connected
    def test_collection_flexible_int32(self):
        self.client.drop_database("bsonnumpy_test")
        docs = [{"x": i, "y": 10-i} for i in range(10)]
        print("docs", docs)
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        ndarray = bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        self.compare_results(
            dtype, self.client.bsonnumpy_test.coll.find(), ndarray)

    @client_context.require_connected
    def test_collection_flexible_mixed(self):
        self.client.drop_database("bsonnumpy_test")
        docs = [{"x": i, "y": random.choice(string.ascii_lowercase)*11} for i in range(10)]
        print("docs", docs)
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype([('x', np.int32), ('y', 'S11')])
        ndarray = bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        self.compare_results(dtype, self.client.bsonnumpy_test.coll.find(), ndarray)

    # @client_context.require_connected
    # def test_collection_standard(self):
    #     self.client.drop_database("bsonnumpy_test")
    #     self.client.bsonnumpy_test.coll.insert(
    #         [{"x": i} for i in range(1000)])
    #     raw_coll = self.client.get_database(
    #         'bsonnumpy_test',
    #         codec_options=CodecOptions(document_class=RawBSONDocument)).coll
    #     cursor = raw_coll.find()
    #
    #     dtype = np.dtype(np.int)
    #     ndarray = bsonnumpy.collection_to_ndarray(
    #         (doc.raw for doc in cursor), dtype, raw_coll.count())

