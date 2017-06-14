import bson
import bsonnumpy
import numpy as np

from test import client_context, TestToNdarray, unittest


class TestErrors(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    docs = [{"x": i, "y": 10 - i} for i in range(10)]
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    def test_incorrect_arguments(self):
        # Expects iterator, dtype, count

        with self.assertRaisesPattern(TypeError, r'.'):
            bsonnumpy.sequence_to_ndarray(None, None, None)
        with self.assertRaisesPattern(TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(0, 0, 0)
        with self.assertRaisesPattern(TypeError, r'sequence_to_ndarray requires a numpy.dtype'):
            bsonnumpy.sequence_to_ndarray(self.docs, None, 10)
        with self.assertRaisesPattern(TypeError, r'function takes exactly 3 arguments \(4 given\)'):
            bsonnumpy.sequence_to_ndarray(self.dtype, self.docs, 10, 10)


    @client_context.require_connected
    def test_incorrect_sequence(self):

        # Non-iterator sequence
        with self.assertRaisesPattern(TypeError, r'sequence_to_ndarray requires an iterator'):
            bsonnumpy.sequence_to_ndarray(None, self.dtype, 10)

        # Empty iterator
        for empty in [[], {}, ()]:
            res = bsonnumpy.sequence_to_ndarray(empty, self.dtype, 10)
            self.assertEqual(res.dtype, self.dtype)
            self.assertEqual(res.size, 0)

        # Non-BSON documents
        with self.assertRaisesPattern(bsonnumpy.error, r'document from sequence failed validation'):
            bsonnumpy.sequence_to_ndarray(self.docs, self.dtype, 10)
        with self.assertRaisesPattern(bsonnumpy.error, r'document from sequence failed validation'):
            bsonnumpy.sequence_to_ndarray((None for _ in range(10)), self.dtype, 10)
        with self.assertRaisesPattern(bsonnumpy.error, r'document from sequence failed validation'):
            bsonnumpy.sequence_to_ndarray(({} for _ in range(10)), self.dtype, 10)


    # def test_incorrect_dtype(self):
    #     print bsonnumpy.sequence_to_ndarray(None, dtype, 10)
    #     # Dtype is named, but does not match documents
    #     # Dtype is not named
    #     # Dtype is null or empty
    #     pass
    #
    # def test_incorrect_count(self):
    #     # Count is greater than iterator
    #     # Count is smaller than iterator
    #     # Count is negative
    #     # Count is null
    #     pass
    #
    # def test_sub_named_fields(self):
    #     # dtype.fields is not a dict
    #     # "expected list from dtype, got other type"
    #     # "expected key from dtype in document, not found"
    #     pass
    #
    # def test_named_scalar_load(self):
    #     # within named dtype, scalar value has sub_dtype->elsize
    #     # within named dtype, scalar value does not match dtype specificed key ("document does not match dtype")
    #     pass
    #
    # def test_sub_array(self):
    #     # dtype expects list, document does not contain array ("expected list from dtype, got another type")
    #     # "expected subarray, got other type"
    #     # "key from dtype not found"
    #     pass
    #
    # def test_array_scalar_load(self):
    #     # array within named dtype contains named dtype
    #     # array within named dtype contains array
    #     # TODO: unhandled case
    #     pass

    # def test_mismatched_dtype1(self):
    #     docs = [{"x": [1, i ,3], "y": 10 - i} for i in range(10)]
    #     dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    #
    #     coll = self.get_cursor_sequence(docs)
    #
    #     bsonnumpy.sequence_to_ndarray(
    #         (doc.raw for doc in coll.find()), dtype, coll.count())
    #
    # def test_mismatched_dtype1(self):
    #     docs = [{"y": 10 - i} for i in range(10)]
    #     dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    #
    #     coll = self.get_cursor_sequence(docs)
    #
    #     bsonnumpy.sequence_to_ndarray(
    #         (doc.raw for doc in coll.find()), dtype, coll.count())
    #
    # def test_unnamed_dtype(self):
    #     docs = [{"y": 10 - i} for i in range(10)]
    #     dtype = np.dtype('10int32')
    #
    #     coll = self.get_cursor_sequence(docs)
    #
    #     bsonnumpy.sequence_to_ndarray(
    #         (doc.raw for doc in coll.find()), dtype, coll.count())
    #
    #
    # def test_wrong_count(self):
    #     docs = [{"x": i, "y": 10 - i} for i in range(10)]
    #     dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    #
    #     coll = self.get_cursor_sequence(docs)
    #     ndarray = bsonnumpy.sequence_to_ndarray(
    #         (doc.raw for doc in coll.find()), dtype, 8)
    #     print ndarray
    #
    #
    #     coll = self.get_cursor_sequence(docs)
    #     ndarray = bsonnumpy.sequence_to_ndarray(
    #         (doc.raw for doc in coll.find()), dtype, 12)
    #     print ndarray
