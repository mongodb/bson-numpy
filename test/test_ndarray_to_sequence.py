import datetime
import math
import random
import string

import bson
import bsonnumpy
import numpy as np
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument

from test import client_context, millis, unittest, TestToNdarray


class TestSequenceFlat(TestToNdarray):
    def test_incorrect_arguments(self):
        # Expects iterator, dtype, count
        needs_iter = r"sequence_to_ndarray requires an iterator"
        invalid = r"document from sequence failed validation"

        with self.assertRaisesPattern(TypeError, needs_iter):
            bsonnumpy.sequence_to_ndarray(1, np.dtype([("a", np.int)]), 1)

        with self.assertRaisesPattern(bsonnumpy.error, invalid):
            bsonnumpy.sequence_to_ndarray("asdf", np.dtype([("a", np.int)]), 1)

        # TODO: better error here
        with self.assertRaisesPattern(bsonnumpy.error, invalid):
            bsonnumpy.sequence_to_ndarray(b"asdf", np.dtype([("a", np.int)]), 1)

        with self.assertRaises(TypeError):
            bsonnumpy.sequence_to_ndarray(10, 10, 1)

        with self.assertRaisesPattern(
                TypeError, "sequence_to_ndarray requires an iterator"):
            bsonnumpy.sequence_to_ndarray(None, np.dtype([("a", np.int)]), 1)

    def test_empty(self):
        dtype = np.dtype([("a", np.int)])
        result = bsonnumpy.sequence_to_ndarray([], dtype, 0)
        self.assertEqual(result.dtype, dtype)
        self.assertTrue(np.array_equal(result, np.array([], dtype)))

    @client_context.require_connected
    def test_int32(self):
        docs = [{"x": i, "y": 10 - i} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int32), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_int64(self):
        docs = [{"x": i, "y": 2**63 - 1 - i} for i in range(10)]
        dtype = np.dtype([('x', np.int64), ('y', np.int64)])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', np.int64), ('x', np.int64)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_objectid(self):
        docs = [{"x": bson.ObjectId()} for _ in range(10)]
        dtype = np.dtype([('x', '<V12')])

        self.client.bsonnumpy_test.coll.delete_many({})
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll

        cursor = raw_coll.find()
        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())

        for i, row in enumerate(ndarray):
            document = docs[i]
            self.assertEqual(document["x"].binary, row["x"].tobytes())

    @client_context.require_connected
    def test_bool(self):
        docs = [{"x": True}, {"x": False}]
        dtype = np.dtype([('x', np.bool)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_datetime(self):
        docs = [{"x": datetime.datetime(1970, 1, 1)},
                {"x": datetime.datetime(1980, 1, 1)},
                {"x": datetime.datetime(1990, 1, 1)}]
        dtype = np.dtype([('x', np.int64)])

        self.client.bsonnumpy_test.coll.delete_many({})
        self.client.bsonnumpy_test.coll.insert_many(docs)
        raw_coll = self.client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll

        cursor = raw_coll.find()
        ndarray = bsonnumpy.sequence_to_ndarray(
            (doc.raw for doc in cursor), dtype, raw_coll.count())

        for i, row in enumerate(ndarray):
            document = docs[i]
            self.assertEqual(
                millis(document["x"] - datetime.datetime(1970, 1, 1)),
                row["x"])

    @client_context.require_connected
    def test_double(self):
        docs = [{"x": math.pi}, {"x": math.pi ** 2}]
        dtype = np.dtype([('x', np.double)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_binary(self):
        docs = [{"x": bson.Binary(b"asdf")}]
        dtype = np.dtype([('x', np.dtype("<V10"))])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_mixed_scalar(self):
        docs = [{"x": i, "y": random.choice(string.ascii_lowercase) * 11} for i
                in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S11'), ('x', np.int32)])
        self.make_mixed_collection_test(docs, dtype)

    def test_void(self):
        # TODO: test for types that are 'V'
        pass


class TestSequenceArray(TestToNdarray):
    @client_context.require_connected
    def test_subarray1d(self):
        # 1d subarray
        docs = [{"x": [1 + i, -i - 1], "y": [i, -i]} for i in range(5)]
        dtype = np.dtype([('x', '2int32'), ('y', '2int32')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', '2int32'), ('x', '2int32')])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_subarray2d(self):
        # 2d subarray
        docs = [{"x": [[i + 0, i + 1, i + 2],
                       [i + 3, i + 4, i + 5],
                       [i + 6, i + 7, i + 8],
                       [i + 9, i + 10, i + 11]],
                 "y": "string!!" + str(i)} for i in range(10)]
        dtype = np.dtype([('x', "(4,3)int32"), ('y', 'S10')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S10'), ('x', "(4,3)int32")])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_subarray3d(self):
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
    def test_subarray2d2(self):
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
    def test_mixed(self):
        docs = [{"x": [i, -i], "y": random.choice(string.ascii_lowercase) * 11,
                 "z": bson.Binary(b'foobar')} for i in range(10)]
        dtype = np.dtype([('x', '2int32'), ('y', 'S11'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('z', 'V12'), ('x', '2int32'), ('y', 'S11')])
        self.make_mixed_collection_test(docs, dtype)
        dtype = np.dtype([('y', 'S11'), ('x', '2int32'), ('z', 'V12')])
        self.make_mixed_collection_test(docs, dtype)


class TestSequenceDoc(TestToNdarray):
    @client_context.require_connected
    def test_subdoc1(self):
        # nested documents
        docs = [{'x': {'y': 100 + i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32)])
        dtype_sub = np.dtype([('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_subdoc2(self):
        # sub-doc has multiple fields
        docs = [{'x': {'y': 100 + i, 'z': i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32), ('z', np.int32)])
        dtype_sub = np.dtype([('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_subdoc3(self):
        # doc has multiple fields
        docs = [{'x': {'y': 100 + i}, 'q': {'y': -i}} for i in range(10)]
        dtype = np.dtype([('y', np.int32)])
        dtype_sub = np.dtype([('x', dtype), ('q', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_subdoc4(self):
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
    def test_subdoc4_mixed(self):
        docs = [{'x': {'y': str(10 + i) * i, 'z': i},
                 'q': {'y': str(i) * i, 'z': 100 - i}} for i in range(10)]
        dtype = np.dtype([('y', 'S110'), ('z', np.int32)])
        dtype_sub = np.dtype([('x', dtype), ('q', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('z', np.int32), ('y', 'S110')])
        dtype_sub = np.dtype([('q', dtype), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_subdoc5(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': 100 + i}}} for i in range(10)]
        dtype0 = np.dtype([('z', np.int32)])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_subdoc6(self):
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


class TestSequenceNestedArray(TestToNdarray):
    def test_nested_array(self):
        docs = [
            {'x': {'y': [100 + i, 100, i],
                   'y1': (i + 1) * 10},
             'x1': i + 5}
            for i in range(10)]

        dtype = np.dtype([('y', '3int32'), ('y1', 'int32')])
        dtype_sub = np.dtype([('x', dtype), ('x1', 'int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y1', 'int32'), ('y', '3int32')])
        dtype_sub = np.dtype([('x1', 'int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_nested_array2x(self):
        docs = [{'x': {'y': [[100 + i, 100, i],
                             [i, i + 1, i + 2]],
                       'y1': (i + 1) * 10},
                 'x1': i + 5} for i in range(10)]
        dtype = np.dtype([('y', '(2,3)int32'), ('y1', 'int32')])
        dtype_sub = np.dtype([('x', dtype), ('x1', 'int32')])
        self.make_mixed_collection_test(docs, dtype_sub)
        dtype = np.dtype([('y1', 'int32'), ('y', '(2,3)int32')])
        dtype_sub = np.dtype([('x1', 'int32'), ('x', dtype)])
        self.make_mixed_collection_test(docs, dtype_sub)

    @client_context.require_connected
    def test_nested_array2x_mixed(self):
        docs = [{'x': {'y': [[100 + i, 100, i],
                             [i, i + 1, i + 2]],
                       'y1': (i + 1) * 10},
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
    def test_nested_array2x_mixed2(self):
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
    def test_nested_array3x(self):
        # 3x nested documents
        docs = [{'x': {'y': {'z': [
            100 + i, 100 - i]}}} for i in range(10)]
        dtype0 = np.dtype([('z', '2int32')])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_nested_array3x2d(self):
        # 3x nested documents with 2d array
        docs = [
            {'x': {'y': {
                'z': [[100 + i, 100 - i, 100],
                      [1 * i, 2 * i, 3 * i],
                      [4 * i, 5 * i, 6 * i],
                      [7 * i, 8 * i, 9 * i]]}}} for i in range(10)]
        dtype0 = np.dtype([('z', '(4,3)int32')])
        dtype1 = np.dtype([('y', dtype0)])
        dtype = np.dtype([('x', dtype1)])
        self.make_mixed_collection_test(docs, dtype)

    @client_context.require_connected
    def test_nested_array3x2d_mixed(self):
        # 3x nested documents with 2d array and other fields
        docs = [
            {'x': {'y': {
                'z': [[100 + i, 100 - i, 100],
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
    def test_nested_array_complicated(self):
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
