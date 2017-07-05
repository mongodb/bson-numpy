import bson
import bsonnumpy
import datetime
import numpy as np
import math
import random
import string

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from test import TestToSequence, millis


class TestFlat(TestToSequence):

    def test_incorrect_arguments(self):
        with self.assertRaisesPattern(
                bsonnumpy.error, r'sequence_to_ndarray requires a numpy\.'
                                 r'ndarray'):
            bsonnumpy.ndarray_to_sequence(None)
        with self.assertRaisesPattern(
                bsonnumpy.error, r'sequence_to_ndarray requires a numpy\.'
                                 r'ndarray'):
            bsonnumpy.ndarray_to_sequence([])
        with self.assertRaisesPattern(
                bsonnumpy.error, r'sequence_to_ndarray requires a numpy\.'
                                 r'ndarray'):
            bsonnumpy.ndarray_to_sequence(1)
        with self.assertRaises(TypeError):
            bsonnumpy.ndarray_to_sequence(10, 10)

    def test_incorrect_ndarray(self):
        with self.assertRaisesPattern(
                bsonnumpy.error, r'ndarray items require named fields'):
            bsonnumpy.ndarray_to_sequence(np.array([i for i in range(10)]))

    def test_empty(self):
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        ndarray = np.array([], dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results([], result, dtype)

    def test_int32(self):
        docs =[{u'x': i, u'y': -i} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        ndarray = np.array([(i, -i) for i in range(10)], dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_int64(self):
        docs = [{"x": i, "y": 2 ** 63 - 1 - i} for i in range(10)]
        dtype = np.dtype([('x', np.int64), ('y', np.int64)])
        ndarray = np.array([(i, 2 ** 63 - 1 - i) for i in range(10)],
                           dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_objectid(self):
        docs = [{"x": bson.ObjectId()} for _ in range(10)]
        dtype = np.dtype([('x', '<V12')])
        ndarray = np.zeros(10, dtype=dtype)
        for i in range(len(docs)):
            ndarray[i]["x"] = docs[i]["x"].binary

        result = bsonnumpy.ndarray_to_sequence(ndarray)
        dict_result = [bson._bson_to_dict(
            d, bson.DEFAULT_CODEC_OPTIONS) for d in result]

        self.assertEqual(len(docs), len(dict_result))
        for i in range(len(docs)):
            self.assertEqual(bson.Binary(docs[i]['x'].binary),
                             dict_result[i]['x'])

    def test_bool(self):
        docs = [{"x": True}, {"x": False}]
        dtype = np.dtype([('x', np.bool)])
        ndarray = np.zeros(2, dtype=dtype)
        ndarray[0]['x'] = True
        ndarray[1]['x'] = False
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_datetime(self):
        docs = [{"x": datetime.datetime(1970, 1, 1)},
                {"x": datetime.datetime(1980, 1, 1)},
                {"x": datetime.datetime(1990, 1, 1)}]
        dtype = np.dtype([('x', np.int64)])
        ndarray = np.zeros(3, dtype)
        for i in range(len(docs)):
            document = docs[i]
            ndarray[i]['x'] = millis(
                document["x"] - datetime.datetime(1970, 1, 1))

        result = bsonnumpy.ndarray_to_sequence(ndarray)

        dict_result = [bson._bson_to_dict(
            d, bson.DEFAULT_CODEC_OPTIONS) for d in result]

        self.assertEqual(len(dict_result), len(docs))

        for i in range(len(dict_result)):
            milli = millis(docs[i]["x"] - datetime.datetime(1970, 1, 1))
            self.assertEqual(
                milli,
                dict_result[i]["x"])

    def test_double(self):
        docs = [{"x": math.pi}, {"x": math.pi ** 2}]
        dtype = np.dtype([('x', np.double)])
        ndarray = np.zeros(2, dtype=dtype)
        ndarray[0]['x'] = math.pi
        ndarray[1]['x'] = math.pi ** 2
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_binary(self):
        docs = [{"x": bson.Binary(b"asdf")}, {"x": bson.Binary(b"kj;iuex")}]
        dtype = np.dtype([('x', np.dtype("<V10"))])
        ndarray = np.zeros(2, dtype=dtype)
        ndarray[0]['x'] = bson.Binary(b"asdf")
        ndarray[1]['x'] = bson.Binary(b"kj;iuex")
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_string(self):
        docs = [{"x": "testing this is a string" + str(i),
                 "y": "testing this is another string" + str(i)}
                for i in range (10)]
        dtype = np.dtype([('x', 'S25'), ('y', 'S31')])
        ndarray = np.array([(docs[i]['x'], docs[i]['y']) for i in range(10)],
                           dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)

    def test_mixed_scalar(self):
        docs = [{"x": i, "y": random.choice(string.ascii_lowercase) * 11} for i
                in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', 'S11')])
        ndarray = np.array([(docs[i]['x'], docs[i]['y']) for i in range(10)],
                           dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result, dtype)


class TestArray(TestToSequence):

    def test_subarray1d(self):
        docs = [{"x": [1 + i, -i - 1], "y": [i, -i]} for i in range(5)]
        dtype = np.dtype([('x', '2int32'), ('y', '2int32')])
        ndarray = np.array([(docs[i]['x'], docs[i]['y']) for i in range(5)],
                           dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        dict_result = [bson._bson_to_dict(
            d, bson.DEFAULT_CODEC_OPTIONS) for d in result]
        print dict_result
        # self.compare_results(docs, result, dtype)
