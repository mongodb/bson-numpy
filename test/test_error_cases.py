import bson
import bsonnumpy
import numpy as np

from test import client_context, TestToNdarray, unittest


class TestErrors(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    bson_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in [
        bson.SON([("x", i), ("y", -i)]) for i in range(10)]]
    ndarray = np.array([(i, -i) for i in range(10)], dtype=dtype)
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    def test_incorrect_arguments(self):
        # Expects iterator, dtype, count
        with self.assertRaisesPattern(TypeError, r'\binteger\b'):
            bsonnumpy.sequence_to_ndarray(None, None, None)
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(0, 0, 0)
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires a numpy.dtype'):
            bsonnumpy.sequence_to_ndarray(self.bson_docs, None, 10)
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(self.dtype, self.bson_docs, 10)
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
        with self.assertRaisesPattern(
                bsonnumpy.error, r'document from sequence failed validation'):
            bsonnumpy.sequence_to_ndarray(docs, self.dtype, 10)
        with self.assertRaisesPattern(
                bsonnumpy.error, r'document from sequence failed validation'):
            bsonnumpy.sequence_to_ndarray(
                (None for _ in range(10)), self.dtype, 10)
        with self.assertRaisesPattern(
                bsonnumpy.error, r'document from sequence failed validation'):
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

        # Dtype is null or empty
        with self.assertRaisesPattern(
                TypeError, r'sequence_to_ndarray requires a numpy.dtype'):
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

    # def test_sub_named_fields(self):
    #     # dtype.fields is not a dict
    #     # "expected list from dtype, got other type"
    #     # "expected key from dtype in document, not found"
    #     pass
    #
    # def test_named_scalar_load(self):
    #     # within named dtype, scalar value has sub_dtype->elsize
    #     # within named dtype, scalar value does not match dtype specificed
            # key ("document does not match dtype")
    #     pass
    #
    # def test_sub_array(self):
    #     # dtype expects list, document does not contain array ("expected list
            #  from dtype, got another type")
    #     # "expected subarray, got other type"
    #     # "key from dtype not found"
    #     pass
    #
    # def test_array_scalar_load(self):
    #     # array within named dtype contains named dtype
    #     # array within named dtype contains array
    #     # TODO: unhandled case
    #     pass

    # def test_wrong_count(self):
