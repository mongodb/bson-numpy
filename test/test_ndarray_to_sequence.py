import bson
import bsonnumpy
import numpy as np

from test import unittest, TestToSequence


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
        self.compare_results([], result)

    def test_int32(self):
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        docs =[bson.SON([("x", i), ("y", -i)]) for i in range(5)]
        ndarray = np.array([(i, -i) for i in range(5)], dtype=dtype)
        result = bsonnumpy.ndarray_to_sequence(ndarray)
        self.compare_results(docs, result)
