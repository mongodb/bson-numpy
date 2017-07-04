import bson
import bsonnumpy
import numpy as np

from test import TestToNdarray, unittest


class TestNdarrayFlat(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    bson_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in [
        bson.SON([("x", i), ("y", -i)]) for i in range(5)]]
    ndarray = np.array([(i, -i) for i in range(5)], dtype=dtype)
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

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


    def test_simple(self):
        result = bsonnumpy.ndarray_to_sequence(self.ndarray)
        print result
        print [bson._bson_to_dict(
            d, bson.DEFAULT_CODEC_OPTIONS) for d in result]
        # bsonnumpy.ndarray_to_sequence(np.array([i for i in range(10)]))
        # bsonnumpy.ndarray_to_sequence(np.array([[i, i] for i in range(10)]))
