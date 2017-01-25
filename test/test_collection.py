import random
import string

import bsonnumpy
import numpy as np
from bson import BSON
from bson.codec_options import CodecOptions
from bson.py3compat import b
from bson.raw_bson import RawBSONDocument

from test import client_context, unittest


class TestCollection2Ndarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = client_context.client

    def compare_elements(self, expected, actual, dtype):
        if isinstance(expected, dict):
            for key, value in expected.items():
                self.compare_elements(value, actual[key],
                                      dtype=dtype.fields[key][0])

        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))

            # If an array's shape is (3,2), its subarrays' shapes are (2,).
            subdtype, shape = dtype.subdtype
            self.assertGreaterEqual(len(shape), 1)
            subarray_dtype = np.dtype((subdtype, shape[1:]))
            for i in range(len(expected)):
                self.compare_elements(expected[i], actual[i], subarray_dtype)

        elif dtype.kind == 'V':
            self.assertEqual(bytes(expected.ljust(dtype.itemsize, b'\0')),
                             bytes(actual))

        elif dtype.kind == 'S':
            # NumPy only stores bytes, not str.
            self.assertEqual(expected, actual.decode('utf-8'))

        else:
            self.assertEqual(expected, actual)

    # TODO: deal with both name and title in dtype
    def compare_results(self, dtype, expected, actual):
        self.assertEqual(dtype, actual.dtype)
        self.assertEqual(expected.count(), len(actual))
        for act in actual:
            exp = next(expected)
            for name in dtype.names:
                self.compare_elements(exp[name], act[name],
                                      dtype=dtype.fields[name][0])

    def make_mixed_collection_test(self, docs, dtype):
        self.client.drop_database("bsonnumpy_test")
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())
        self.compare_results(np.dtype(dtype),
                             self.client.bsonnumpy_test.coll.find(),
                             ndarray)

    @client_context.require_connected
    def test_collection_flexible_int32(self):
        docs = [{"x": i, "y": 10 - i} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int32), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_mixed_scalar(self):
        docs = [{"x": i, "y": random.choice(string.ascii_lowercase) * 11} for i
                in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S11'), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray1(self):
        # 2d subarray
        docs = [{"x": [1 + i, -i - 1], "y": [i, -i]} for i in range(5)]
        dtype = np.dtype([('x', '2int32'), ('y', '2int32')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', '2int32'), ('x', '2int32')])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray2(self):
        # 3d subarray
        docs = [{"x": [[i, i + 1, i + 2],
                       [-i, -i - 1, -i - 2],
                       [100 * i, 100 * i + 1, 100 * i + 2],
                       [0, 1, 2]],
                 "y": "string!!!"} for i in range(5)]
        dtype = np.dtype([('x', "(4,3)int32"), ('y', 'S10')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S10'), ('x', "(4,3)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray3(self):
        # 3d subarray
        docs = []
        for i in range(5):
            docs.append({
                "x": [
                    [
                        [i, i + 1], [i + 1, i + 2], [i + 2, i + 3]
                    ],
                    [
                        [-i, -i + 1], [-i - 1, -i], [-i - 2, -i - 1]
                    ],
                    [
                        [100 * i, 100 * i + i], [100 * i + 1, 100 * i + i],
                        [100 * i + 2, 100 * i + i]
                    ],
                    [
                        [0, 1], [1, 2], [3, 4]
                    ]
                ],
                "some_other_key": [
                    "string" + str(i), "string" + str(i + 1)
                ]
            })

        dtype = np.dtype([('x', "(4,3,2)int32"), ('some_other_key', '2S10')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('some_other_key', '2S10'), ('x', "(4,3,2)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_subarray2_mixed1(self):
        # 3d subarray
        docs = [{"x": [[i, i + 1, i + 2],
                       [-i, -i - 1, -i - 2],
                       [100 * i, 100 * i + 1, 100 * i + 2]],
                 "y": 100 - i} for i in range(2)]
        dtype = np.dtype([('x', "(3,3)int32"), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int32), ('x', "(3,3)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_flexible_mixed(self):
        docs = [{"x": [i, -i], "y": random.choice(string.ascii_lowercase) * 11,
                 "z": bson.Binary(b'foobar')} for i in range(10)]
        dtype = np.dtype([('x', '2int32'), ('y', 'S11'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('z', 'V12'), ('x', '2int32'), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S11'), ('x', '2int32'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub1(self):
        # nested documents
        docs = [{'x': {'y': 100 + i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32)])
        dtype_sub = np.dtype([('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub2(self):
        # sub-doc has multiple fields
        docs = [{'x': {'y': 100 + i, 'z': i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32), ('z', np.int32)])
        dtype_sub = np.dtype([('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub3(self):
        # doc has multiple fields
        docs = [{'x': {'y': 100 + i}, 'q': {'y': -i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32)])
        dtype_sub = np.dtype([('x', dtype), ('q', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub4(self):
        # doc and subdoc have multiple fields
        docs = [{'x': {'y': 100 + i, 'z': i}, 'q': {'y': -i, 'z': 100 - i}} for
                i in range(10)]
        dtype = np.dtype([('y', np.int32), ('z', np.int32)])
        dtype_sub = np.dtype([('x', dtype), ('q', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('z', np.int32), ('y', np.int32)])
        dtype_sub = np.dtype([('q', dtype), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub4_mixed(self):
        docs = [{'x': {'y': str(10 + i) * i, 'z': i},
                 'q': {'y': str(i) * i, 'z': 100 - i}} for i in range(10)]
        dtype = np.dtype([('y', 'S110'), ('z', np.int32)])
        dtype_sub = np.dtype([('x', dtype), ('q', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('z', np.int32), ('y', 'S110')])
        dtype_sub = np.dtype([('q', dtype), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub_subarrays(self):
        docs = [
            {'x': {'y': [100 + i, 100, i], 'y1': (i + 1) * 10}, 'x1': i + 5}
            for i in range(10)]

        dtype = np.dtype([('y', '3int32'), ('y1', 'int32')])
        dtype_sub = np.dtype([('x', dtype), ('x1', 'int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y1', 'int32'), ('y', '3int32')])
        dtype_sub = np.dtype([('x1', 'int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub_subarrays2(self):
        docs = [{'x': {'y': [[100 + i, 100, i],
                             [i, i + 1, i + 2]], 'y1': (i + 1) * 10},
                 'x1': i + 5} for i in range(10)]
        dtype = np.dtype([('y', '(2,3)int32'), ('y1', 'int32')])
        dtype_sub = np.dtype([('x', dtype), ('x1', 'int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y1', 'int32'), ('y', '(2,3)int32')])
        dtype_sub = np.dtype([('x1', 'int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub_subarrays3(self):
        docs = [{'x': {'y': [[100 + i, 100, i],
                             [i, i + 1, i + 2]], 'y1': (i + 1) * 10},
                 'x1': [[[i + 5, i + 6], [i + 7, i + 8]],
                        [[i + 9, i + 10], [i + 11, i + 12]]]} for i in
                range(10)]
        dtype = np.dtype([('y', '(2,3)int32'), ('y1', 'int32')])
        dtype_sub = np.dtype([('x', dtype), ('x1', '(2,2,2)int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y1', 'int32'), ('y', '(2,3)int32')])
        dtype_sub = np.dtype([('x1', '(2,2,2)int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub_subarrays4(self):
        docs = [{'x': {'y': [[100 + i, 100, i],
                             [i, i + 1, i + 2]],
                       'y1': (i + 1) * 10,
                       'y2': random.choice(string.ascii_lowercase) * i},
                 'x1': [[[i + 5, i + 6], [i + 7, i + 8]],
                        [[i + 9, i + 10], [i + 11, i + 12]]]} for i in
                range(10)]
        dtype = np.dtype([('y', '(2,3)int32'), ('y1', 'int32'), ('y2', 'S12')])
        dtype_sub = np.dtype([('x', dtype), ('x1', '(2,2,2)int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y2', 'S12'), ('y1', 'int32'), ('y', '(2,3)int32')])
        dtype_sub = np.dtype([('x1', '(2,2,2)int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_collection_sub4(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': 100 + i}}} for i in range(10)]
        dtype0 = np.dtype([('z', np.int32)])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub4_array(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': [100 + i, 100 - i]}}} for i in range(10)]
        dtype0 = np.dtype([('z', '2int32')])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub4_array(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': [
            [100 + i, 100 - i, 100],
            [1 * i, 2 * i, 3 * i],
            [4 * i, 5 * i, 6 * i],
            [7 * i, 8 * i, 9 * i]]}}} for i in range(10)]
        dtype0 = np.dtype([('z', '(4,3)int32')])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub4_array2(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': [[100 + i, 100 - i, 100],
                                   [1 * i, 2 * i, 3 * i],
                                   [4 * i, 5 * i, 6 * i],
                                   [7 * i, 8 * i, 9 * i]],
                             'z2': "this is a string!"},
                       'y2': {'a': "a different doc string"}},
                 'x2': [1, 2, 3]} for i in range(10)]
        dtype2 = np.dtype([('a', 'S26')])
        dtype0 = np.dtype([('z', '(4,3)int32'), ('z2', 'S17')])
        dtype1 = np.dtype([('y', dtype0), ('y2', dtype2)])
        dtype = np.dtype([('x', dtype1), ('x2', '3int32')])
        self.make_mixed_collection_test(docs, dtype)
        dtype0 = np.dtype([('z2', 'S17'), ('z', '(4,3)int32')])
        dtype1 = np.dtype([('y2', dtype2), ('y', dtype0)])
        dtype = np.dtype([('x2', '3int32'), ('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub4_array3(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': i,
                             'z2': "this is a string!"},
                       'y2': i},
                 'x2': i} for i in range(10)]
        dtype0 = np.dtype([('z', np.int32), ('z2', 'S17')])
        dtype1 = np.dtype([('y', dtype0), ('y2', np.int32)])
        dtype = np.dtype([('x', dtype1), ('x2', 'int32')])
        self.make_mixed_collection_test(docs, dtype)

        dtype0 = np.dtype([('z2', 'S17'), ('z', np.int32)])
        dtype1 = np.dtype([('y2', np.int32), ('y', dtype0)])
        dtype = np.dtype([('x2', 'int32'), ('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_sub_many(self):
        num = 3
        docs = [{} for _ in range(num)]

        for i in range(num):
            doc = docs[i]
            letter_index = 2
            for letter in string.ascii_lowercase:
                doc[letter] = {}
                subarray = [[letter for _ in range(letter_index)] for _ in
                            range(letter_index)]
                doc[letter + '1'] = subarray
                doc[letter + '2'] = letter * random.randint(0, 100)
                doc = doc[letter]
                letter_index += 1
            doc['LOWEST'] = [1 * i + 99, 2 * i + 98, 3 * i + 97]
            doc['LOWEST2'] = i
            doc['LOWEST3'] = 'another long string'

        dt = np.dtype(
            [('LOWEST', '3int32'), ('LOWEST2', np.int32), ('LOWEST3', 'S20')])
        letter_index = len(string.ascii_lowercase) + 1
        for letter in string.ascii_lowercase[::-1]:
            type_name = '(' + str(letter_index) + ',' + str(
                letter_index) + ')S2'
            dt = np.dtype([(letter, dt), (letter + '2', 'S100'),
                           (letter + '1', type_name)])
            letter_index -= 1

        self.make_mixed_collection_test(docs, dt)  # OMG this works!!

    @client_context.require_connected
    def test_collection_not_flexible(self):
        # TODO: determine what to do when user doesn't give a flexible type for
        # documents. Doc order?
        docs = [{"x": [i, i - 1], "y": [10 - i, 9 - i]} for i in range(10)]
        dtype = np.dtype("(2, 2)int32")
        # self.make_mixed_collection_test(docs, dtype)
        docs = [{"x": i, "y": 10 - i} for i in range(10)]
        dtype = np.dtype("2int32")
        # self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_collection_wrong_count(self):
        dtype = np.dtype([('_id', np.int32)])
        docs = [{"_id": 1}, {"_id": 2}]
        self.client.bsonnumpy_test.coll.delete_many({})
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll

        cursor = raw_coll.find()
        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, 2)

        self.assertEqual(2, len(ndarray))
        cursor.rewind()

        # Undercount.
        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, 1)

        self.assertEqual(1, len(ndarray))
        cursor.rewind()

        # Overcount.
        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, 30)

        self.assertEqual(2, len(ndarray))
