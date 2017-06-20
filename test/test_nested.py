import bson
import bsonnumpy
import numpy as np

from test import client_context, TestToNdarray, unittest


class TestNested(TestToNdarray):
    dtype = np.dtype([('x', np.int32), ('y', np.int32)])
    bson_docs = [bson._dict_to_bson(
        doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in [
        bson.SON([("x", i), ("y", -i)]) for i in range(10)]]
    ndarray = np.array([(i, -i) for i in range(10)], dtype=dtype)
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    # @unittest.skip("not yet implemented")
    def test_array_scalar_load00(self):
        # Test arrays with documents as elements

        son_docs = [
            bson.SON([('x', [i, i, i, i])]) for i in range(2, 4)]
        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        dtype = np.dtype([('x', '4int32')])

        ndarray = np.array([([i, i, i, i],) for i in range(2, 4)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 2)

        print "expected", ndarray
        print "actual", res
        self.assertTrue(np.array_equal(ndarray, res))

    def test_array_scalar_load0(self):
        # Test arrays with documents as elements

        son_docs = [
            bson.SON([('x', [[i, i] for _ in range(3)])]) for i in range(2, 4)]
        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        dtype = np.dtype([('x', '(3,2)int32')])

        ndarray = np.array(
            [([[i, i] for _ in range(3)],) for i in range(2, 4)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 2)

        print "expected", ndarray
        print "actual", res
        self.assertTrue(np.array_equal(ndarray, res))

    # @unittest.skip("not yet implemented")
    def test_array_scalar_load1(self):
        # Test arrays with documents as elements

        son_docs = [
            bson.SON(
                [('x', [
                    bson.SON([('a', i), ('b', i)]),
                    bson.SON([('a', -i), ('b', -i)])
                ])]) for i in range(2, 4)]
        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        sub_dtype = np.dtype(([('a', 'int32'), ('b', 'int32')], 2))
        dtype = np.dtype([('x', sub_dtype)])

        ndarray = np.array([([(i, i), (-i, -i)],) for i in range(2, 4)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 2)
        self.assertTrue(np.array_equal(ndarray, res))

    # @unittest.skip("not yet implemented")
    def test_array_scalar_load2(self):
        # Test sub arrays with documents as elements
        son_docs = [
            bson.SON(
                [('x', [
                    [
                        bson.SON([('a', i), ('b', i)]),
                        bson.SON([('a', -i), ('b', -i)])
                    ],
                    [
                        bson.SON([('c', i), ('d', i)]),
                        bson.SON([('c', -i), ('d', -i)])
                    ],

                ])]) for i in range(2, 4)]
        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        sub_sub_dtype = np.dtype(([('a', 'int32'), ('b', 'int32')], 2))
        sub_dtype = np.dtype((sub_sub_dtype, 2))
        dtype = np.dtype([('x', sub_dtype)])

        ndarray = np.array(
            [[([(i, i), (-i, -i)],),
              ([(i, i), (-i, -i)],)] for i in range(2, 4)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 2)
        print "expected", ndarray
        print "actual", res
        self.assertTrue(np.array_equal(ndarray, res))

    @unittest.skip("not yet implemented")
    def test_array_scalar_load3(self):
        # Test sub arrays with documents that have arrays
        son_docs = [
            bson.SON(
                [('x', [
                    bson.SON([('a', [i, i, i, i]),
                              ('b', [i, i, i, i])]),
                    bson.SON([('a', [-i, -i, -i, -i]),
                              ('b', [-i, -i, -i, -i])])
                ])]) for i in range(10)]

        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        sub_dtype = np.dtype(([('a', '4int32'), ('b', '4int32')], 2))
        dtype = np.dtype([('x', sub_dtype)])

        ndarray = np.array(
            [([([i, i, i, i], [i, i, i, i]),
               ([-i, -i, -i, -i], [-i, -i, -i, -i])],)
             for i in range(10)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 4)
        self.assertTrue(np.array_equal(ndarray, res))

    @unittest.skip("not yet implemented")
    def test_array_scalar_load4(self):
        # Test documents with multiple levels of sub documents
        son_docs = [
            bson.SON(
                [('x', [
                    [
                        bson.SON([('a', i), ('b', i)]),
                        bson.SON([('a', -i), ('b', -i)])
                    ],
                    [
                        bson.SON([('c', i), ('d', i)]),
                        bson.SON([('c', -i), ('d', -i)])
                    ],

                ])]) for i in range(10)]
        raw_docs = [bson._dict_to_bson(
            doc, False, bson.DEFAULT_CODEC_OPTIONS) for doc in son_docs]
        sub_sub_sub_dtype = np.dtype([('q', 'int32')])
        sub_sub_dtype = np.dtype(
            ([('a', sub_sub_sub_dtype), ('b', sub_sub_sub_dtype)], 2))
        sub_dtype = np.dtype((sub_sub_dtype, 2))
        dtype = np.dtype([('x', sub_dtype)])

        ndarray = np.array([([[((i,), (i,)), ((-i,), (-i,))],
                              [((i,), (i,)), ((-i,), (-i,))]],)
                            for i in range(10)], dtype)

        # Correct dtype
        res = bsonnumpy.sequence_to_ndarray(raw_docs, dtype, 4)
        self.assertTrue(np.array_equal(ndarray, res))
