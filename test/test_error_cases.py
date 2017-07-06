import bson
import bsonnumpy
import numpy as np

from test import TestToNdarray, unittest


class TestErrors(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    bson_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in [
        bson.SON([("x", i), ("y", -i)]) for i in range(10)]]
    ndarray = np.array([(i, -i) for i in range(10)], dtype=dtype)

    def test_incorrect_arguments(self):
        # Expects iterator, dtype, count
        with self.assertRaisesPattern(TypeError, r'\binteger\b'):
            bsonnumpy.sequence_to_ndarray(None, None, None)
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(0, self.dtype, 0)
        with self.assertRaisesPattern(
                bsonnumpy.error, r'dtype must include field names'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, None, 10)
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(self.dtype, self.dtype, 10)
        with self.assertRaisesPattern(
                TypeError, r'function takes exactly 3 arguments \(4 given\)'):
            bsonnumpy.sequence_to_ndarray(self.dtype, self.bson_docs, 10, 10)

    def test_incorrect_sequence(self):
        docs = [{"x": i, "y": -i} for i in range(10)]

        # Non-iterator sequence
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(None, self.dtype, 10)

        # Empty iterator
        for empty in [[], {}, ()]:
            res = bsonnumpy.sequence_to_ndarray(empty, self.dtype, 10)
            self.assertEqual(res.dtype, self.dtype)
            self.assertEqual(res.size, 0)

        # Non-BSON documents
        msg = r'sequence_to_ndarray requires sequence of bytes objects'
        with self.assertRaisesPattern(TypeError, msg):
            bsonnumpy.sequence_to_ndarray(docs, self.dtype, 10)
        with self.assertRaisesPattern(TypeError, msg):
            bsonnumpy.sequence_to_ndarray(
                (None for _ in range(10)), self.dtype, 10)
        with self.assertRaisesPattern(TypeError, msg):
            bsonnumpy.sequence_to_ndarray(
                ({} for _ in range(10)), self.dtype, 10)

    def test_incorrect_dtype(self):
        dtype = np.dtype([('a', np.int32), ('b', np.int32)])

        # Dtype is named, but does not match documents
        with self.assertRaisesPattern(
                bsonnumpy.error, r'document does not match dtype'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, dtype, 10)

        # Dtype is not named
        with self.assertRaisesPattern(
                bsonnumpy.error,
                r'dtype must include field names,'
                r' like dtype\(\[\(\'fieldname\', numpy.int\)\]\)'):
            bsonnumpy.sequence_to_ndarray(
                self.bson_docs, np.dtype(np.int32), 10)

        # Dtype is simple array
        with self.assertRaisesPattern(
                bsonnumpy.error,
                r'dtype must include field names,'
                r' like dtype\(\[\(\'fieldname\', numpy.int\)\]\)'):
            bsonnumpy.sequence_to_ndarray(
                self.bson_docs, np.dtype('(3,2)int32'), 10)

        # Dtype is null or empty
        with self.assertRaisesPattern(
                bsonnumpy.error, r'dtype must include field names'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, None, 1)

    def test_incorrect_count(self):
        self.assertTrue(
            np.array_equal(
                bsonnumpy.sequence_to_ndarray(self.bson_docs, self.dtype, 100),
                self.ndarray))
        self.assertTrue(
            np.array_equal(
                bsonnumpy.sequence_to_ndarray(self.bson_docs, self.dtype, 5),
                self.ndarray[:5]))
        with self.assertRaisesPattern(
                bsonnumpy.error, r'count argument was negative'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, self.dtype, -10)
        with self.assertRaisesPattern(TypeError, r'\binteger\b'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, self.dtype, None)

    def test_null(self):
        data = bson._dict_to_bson(
            {"x": None}, True, bson.DEFAULT_CODEC_OPTIONS)
        with self.assertRaisesPattern(bsonnumpy.error,
                                      r'unsupported BSON type: Null'):
            bsonnumpy.sequence_to_ndarray(iter([data]),
                                          np.dtype([('x', '<V10')]), 1)

    def test_string_length(self):
        data = bson._dict_to_bson({"x": "abc"}, True,
                                  bson.DEFAULT_CODEC_OPTIONS)

        ndarray = bsonnumpy.sequence_to_ndarray(iter([data]),
                                                np.dtype([("x", "V1")]),
                                                1)

        self.assertEqual(ndarray[0]["x"].tobytes(), b"a")
        ndarray = bsonnumpy.sequence_to_ndarray(iter([data]),
                                                np.dtype([("x", "V2")]),
                                                1)

        self.assertEqual(ndarray[0]["x"].tobytes(), b"ab")
        ndarray = bsonnumpy.sequence_to_ndarray(iter([data]),
                                                np.dtype([("x", "V3")]),
                                                1)

        self.assertEqual(ndarray[0]["x"].tobytes(), b"abc")
        ndarray = bsonnumpy.sequence_to_ndarray(iter([data]),
                                                np.dtype([("x", "V4")]),
                                                1)

        self.assertEqual(ndarray[0]["x"].tobytes(), b"abc\0")


class TestDocErrors(TestToNdarray):

    son_docs = [
        bson.SON(
            [("x", bson.SON([("y", i), ("z", i)])),
             ("q", bson.SON([("y", i), ("z", i)]))]
        ) for i in range(10)]
    raw_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
    dtype = np.dtype([('y', np.int32), ('z', np.int32)])
    dtype_sub = np.dtype([('x', dtype), ('q', dtype)])

    ndarray = np.array([((i, i), (i, i)) for i in range(10)],
                       dtype=dtype_sub)

    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    def test_correct_sub_dtype(self):
        res = bsonnumpy.sequence_to_ndarray(self.raw_docs, self.dtype_sub, 10)
        np.testing.assert_array_equal(self.ndarray, res)

    def test_incorrect_sub_dtype1(self):
        # Top document missing key
        bad_doc = bson.SON(
            [("bad", bson.SON([("y", 0), ("z", 0)])),
             ("q", bson.SON([("y", 0), ("z", 0)]))])

        bad_raw_docs = self.raw_docs[:9]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(bsonnumpy.error,
                                      "document does not match dtype"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype_sub, 10)

    def test_incorrect_sub_dtype2(self):
        # Top document has extra key
        data = bson._dict_to_bson({"x": 12, "y": 13}, True,
                                  bson.DEFAULT_CODEC_OPTIONS)

        ndarray = bsonnumpy.sequence_to_ndarray(
            [data], np.dtype([("y", np.int)]), 1)

        self.assertEqual(1, len(ndarray))
        self.assertEqual(13, ndarray[0]["y"])

        with self.assertRaises(ValueError):
            ndarray[0]["x"]

    def test_incorrect_sub_dtype3(self):
        # Sub document missing key
        bad_doc = bson.SON(
            [("x", bson.SON([("bad", 0), ("z", 0)])),
             ("q", bson.SON([("y", 0), ("z", 0)]))])

        bad_raw_docs = self.raw_docs[:9]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(bsonnumpy.error,
                                      "document does not match dtype"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype_sub, 10)

    def test_incorrect_sub_dtype4(self):
        # Sub document not a document
        bad_doc = bson.SON(
            [("x", bson.SON([("y", 0), ("z", 0)])),
             ("q", 10)])

        bad_raw_docs = self.raw_docs[:9]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: expected subdoc from dtype,"
                " got other type"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype_sub, 10)

        bad_doc = bson.SON(
            [("x", bson.SON([("y", 0), ("z", 0)])),
             ("q", [10, 11, 12])])

        bad_raw_docs = self.raw_docs[:9]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: expected subdoc from dtype,"
                " got other type"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype_sub, 10)

    def test_incorrect_sub_dtype5(self):
        # Sub document extra key
        dtype2 = np.dtype([('y', np.int32), ('z', np.int32)])
        dtype_sub2 = np.dtype([('x', dtype2)])

        ndarray2 = np.array([((i, i),) for i in range(10)], dtype=dtype_sub2)
        res = bsonnumpy.sequence_to_ndarray(self.raw_docs, dtype_sub2, 10)
        self.assertTrue(np.array_equal(ndarray2, res))

        dtype3 = np.dtype([('y', np.int32)])
        dtype_sub3 = np.dtype([('x', dtype3), ('q', dtype3)])
        ndarray3 = np.array([((i,), (i,)) for i in range(10)],
                            dtype=dtype_sub3)
        res = bsonnumpy.sequence_to_ndarray(self.raw_docs, dtype_sub3, 10)
        self.assertTrue(np.array_equal(ndarray3, res))


class TestArrayErrors(TestToNdarray):

    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    son_docs = [
        bson.SON(
            [("x", [[i, i*2, i*3], [i*4, i*5, i*6]]),
             ("y", [[i*7, i*8, i*9], [i*10, i*11, i*12]])])
        for i in ['a', 'b', 'c', 'd']]
    raw_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
    dtype = np.dtype([('x', '2,3S13'), ('y', '2,3S13')])

    ndarray = np.array(
        [([[i, i*2, i*3], [i*4, i*5, i*6]],
         ([[i*7, i*8, i*9], [i*10, i*11, i*12]]))
            for i in ['a', 'b', 'c', 'd']], dtype=dtype)

    def test_correct_sub_dtype_array(self):
        res = bsonnumpy.sequence_to_ndarray(self.raw_docs, self.dtype, 4)
        self.assertTrue(np.array_equal(self.ndarray, res))

    def test_incorrect_sub_dtype_array1(self):
        # Top document missing key
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3], ['d'*4, 'd'*5, 'd'*6]]),
             ("bad_key", [['d'*7, 'd'*7, 'd'*9], ['d'*10, 'd'*11, 'd'*12]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))
        with self.assertRaisesPattern(bsonnumpy.error,
                                      "document does not match dtype"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array2(self):
        # Top-level array not array
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3], ['d'*4, 'd'*5, 'd'*6]]),
             ("y", 'not an array')])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: expected list from dtype, got other type"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array3(self):
        # Top-level array too long
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3],
                    ['d'*4, 'd'*5, 'd'*6],
                    ['d'*4, 'd'*5, 'd'*6]]),
             ("y", [['d'*7, 'd'*8, 'd'*9],
                    ['d'*10, 'd'*11, 'd'*12],
                    ['d'*10, 'd'*11, 'd'*12]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))
        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: list is of incorrect length"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array4(self):
        # Sub array too long
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3, 'd'*3],
                    ['d'*4, 'd'*5, 'd'*6, 'd'*3]]),
             ("y", [['d'*7, 'd'*8, 'd'*9, 'd'*3],
                    ['d'*10, 'd'*11, 'd'*12, 'd'*3]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))
        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: list is of incorrect length"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array5(self):
        # Sub array not array
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3], ['d'*4, 'd'*5, 'd'*6]]),
             ("y", ['not an array', ['d'*10, 'd'*11, 'd'*12]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: expected list from dtype, got other type"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array6(self):
        # Top-level array too short
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2, 'd'*3]]),
             ("y", [['d'*7, 'd'*8, 'd'*9]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))

        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: list is of incorrect length"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)

    def test_incorrect_sub_dtype_array7(self):
        # Sub array too short
        bad_doc = bson.SON(
            [("x", [['d'*1, 'd'*2], ['d'*4, 'd'*5]]),
             ("y", [['d'*7, 'd'*8], ['d'*10, 'd'*11]])])
        bad_raw_docs = self.raw_docs[:3]
        bad_raw_docs.append(
            bson._dict_to_bson(bad_doc, False, bson.DEFAULT_CODEC_OPTIONS))
        with self.assertRaisesPattern(
                bsonnumpy.error,
                "invalid document: list is of incorrect length"):
            bsonnumpy.sequence_to_ndarray(bad_raw_docs, self.dtype, 4)
