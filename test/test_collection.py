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

    @client_context.require_connected
    def test_iterator(self):
        self.client.drop_database("bsonnumpy_test")
        self.client.bsonnumpy_test.coll.insert([{"x": i} for i in range(1000)])
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype("int32")
        bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        # print ("NDARRAY", ndarray)

    @client_context.require_connected
    def test_flexible_type(self):
        self.client.drop_database("bsonnumpy_test")
        num_docs = 10
        names = [random.choice(string.lowercase)*10 for i in range(num_docs)]
        self.client.bsonnumpy_test.coll.insert([{"name": names[i],
                                                 "grades": [random.random(),
                                                            random.random()]}
                                                for i in range(num_docs)])
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype([('name', np.str_, 18), ('grades', np.float64, (2,))])
        ndarray = bsonnumpy.collection_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        print ("NDARRAY", ndarray)
