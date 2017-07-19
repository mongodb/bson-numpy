import bson
import bsonnumpy
import numpy as np
from bson import DEFAULT_CODEC_OPTIONS as DEFAULT, SON

from test import TestToNdarray, unittest


class TestNdarrayFlat(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    bson_docs = [
        bson._dict_to_bson(bson.SON([("x", i), ("y", -i)]), False, DEFAULT)
        for i in range(10)]

    ndarray = np.array([(i, -i) for i in range(10)], dtype=dtype)
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

    def test_empty(self):
        dtype = np.dtype([('x', np.int32), ('y', np.float)])
        batch = b''.join([
            bson.BSON.encode({"x": 1, "y": 1.1}),
            bson.BSON.encode({}),
            bson.BSON.encode({"x": 3, "y": 1.3}),
        ])

        with self.assertRaisesPattern(
                bsonnumpy.error, r'document does not match dtype'):
            bsonnumpy.sequence_to_ndarray([batch], dtype, 3)

    def test_raw_batch(self):
        dtype = np.dtype([('x', np.int32), ('y', np.float)])

        # A variety of lengths.
        batch = b''.join([
            bson.BSON.encode({"x": 1, "y": 1.1}),
            bson.BSON.encode({"x": 2, "y": 1.2, "extra key": "foobar"}),
            bson.BSON.encode({"x": 3, "y": 1.3}),
        ])

        result = bsonnumpy.sequence_to_ndarray([batch], dtype, 3)
        ndarray = np.array([(1, 1.1), (2, 1.2), (3, 1.3)], dtype)
        np.testing.assert_array_equal(result, ndarray)

        dtype = np.dtype([('x', np.int32), ('y', np.float), ('z', np.int32)])

        # A variety of orders.
        batch = b''.join([
            bson.BSON.encode(SON([("x", 1), ("y", 1.1), ("z", 4)])),
            bson.BSON.encode(SON([("x", 2), ("z", 5), ("y", 1.2)])),
            bson.BSON.encode(SON([("z", 6), ("x", 3), ("y", 1.3)]))
        ])

        result = bsonnumpy.sequence_to_ndarray([batch], dtype, 3)
        ndarray = np.array([(1, 1.1, 4), (2, 1.2, 5), (3, 1.3, 6)], dtype)
        np.testing.assert_array_equal(result, ndarray)

if __name__ == '__main__':
    unittest.main()
