import random
import string

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
import numpy as np

import bsonnumpy
from test import unittest, client_context


class TestCollection2Ndarray(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = client_context.client

    # TODO: deal with both name and title in dtype

    @client_context.require_connected
    def test_collection_flexible(self):
        self.client.drop_database("bsonnumpy_test")
        docs = [{"x": i, "y": 10-i} for i in range(10)]
        print "docs", docs
        self.client.bsonnumpy_test.coll.insert(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype([('x', np.int), ('y', np.int)])
        ndarray = bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        print ndarray

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

