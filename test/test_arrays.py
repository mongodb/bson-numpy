import bson
import bsonnumpy
import numpy as np

from test import TestFromBSON


class TestFromBSONArrays(TestFromBSON):
    def test_constant(self):
        document = bson.SON([("0", 99),
                             ("1", 88),
                             ("2", 77)])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        # print (result)
        self.assertEqual(dtype, result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

    def test_nd_len1_int(self):
        # arrays of len 1 become constants, except if top-level array is 1.

        document = bson.SON([("0", [99, 88, 77])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        # print (result)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON([("0", [[99, 88, 77]])])
        utf8 = bson._dict_to_bson(
            same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        # print (result)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON([("0", [[99], [88], [77]])])
        utf8 = bson._dict_to_bson(
            same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        # print (result)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON(
            [("0", [[[[[99]]]], [[[[[88]]]]], [[[[77]]]]])])
        utf8 = bson._dict_to_bson(
            same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        # print (result)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

    def test_2d_int(self):
        document = bson.SON([("0", [9, 8]),
                             ("1", [6, 5]),
                             ("2", [3, 2])])  # [ [a,b], [c,d], [e,f] ]
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("2int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

    def test_3d_int(self):
        document = bson.SON([("0", [[9, 9], [8, 8], [7, 7]]),
                             ("1", [[6, 6], [5, 5], [4, 4]]),
                             ("2", [[3, 3], [2, 2], [1, 1]])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('(3,2)int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

    def test_3d_len1(self):
        # arrays of length 1 are maintained when they are within another array
        document = bson.SON([("0",
                              [[[9], [9]],
                               [[8], [8]],
                               [[7], [7]]]),
                             ("1",
                              [[[6], [6]],
                               [[5], [5]],
                               [[4], [4]]]),
                             ("2",
                              [[[3], [3]],
                               [[2], [2]],
                               [[1], [1]]])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('(3,2,1)int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        document = bson.SON([("0", [[9], [8], [7]]),
                             ("1", [[6], [5], [4]]),
                             ("2", [[3], [2], [1]])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('(3,1)int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        expected_document = bson.SON([("0", [7, 7]),
                                      ("1", [4, 4]),
                                      ("2", [1, 1])])
        document = bson.SON([("0", [[7, 7]]),
                             ("1", [[4, 4]]),
                             ("2", [[1, 1]])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('2int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(
                np.array_equal(expected_document[str(b)], result[b]))

        document = bson.SON(
            [("0", [[[[99]]], [[[88]]]])])
        utf8 = bson._dict_to_bson(
            document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("(2,1,1,1)int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))
