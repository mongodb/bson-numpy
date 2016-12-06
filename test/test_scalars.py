import bson
import datetime
import numpy as np
import pymongo
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
import random
import struct
import string


import bsonnumpy


from test import unittest


def millis(delta):
    if hasattr(delta, 'total_seconds'):
        return delta.total_seconds() * 1000

    # Python 2.6.
    return ((delta.days * 86400 + delta.seconds) * 1000 +
            delta.microseconds / 1000.0)


class TestArray2Ndarray(unittest.TestCase):
    def compare_results(self, np_type, document, compare_to):
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype(np_type)
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(result.dtype, dtype)
        for i in range(len(result)):
            self.assertEqual(compare_to[str(i)], result[i],
                             "Comparison failed for type %s: %s != %s" % (
                                 dtype, compare_to[str(i)], result[i]))

    def test_integer32_types(self):
        document = bson.SON([("0", 99), ("1", 88), ("2", 77), ("3", 66)])
        for np_type in [np.int8,
                        np.int16, np.int32,
                        np.uint8, np.uint16, np.uint32
                       ]:
            self.compare_results(np_type, document, document)

    def test_do_nothing(self):
        pass

    def test_integer64_types(self):
        document = bson.SON(
            [("0", 99), ("1", 88), ("2", 77)])
        for np_type in [np.int_, np.intc, np.intp, np.uint64, np.int64]:
            self.compare_results(np_type, document, document)

    def test_bool_types(self):
        document = bson.SON([("0", True), ("1", False), ("2", True)])
        self.compare_results(np.bool_, document, document)

    def test_float64_types(self):
        document = bson.SON(
            [("0", 99.99), ("1", 88.88), ("2", 77.77)])
        for np_type in [np.float_, np.float64]:
            self.compare_results(np_type, document, document)

    def TODO_num_types(self):
        np_types = [np.complex_, np.complex64, np.complex128, np.float32 ] # https://jira.mongodb.org/browse/SERVER-9342

    def test_oid(self):
        document = bson.SON(
            [("0", bson.ObjectId()), ("1", bson.ObjectId()),
             ("2", bson.ObjectId())])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('<V12')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(result.dtype, dtype)
        for i in range(len(result)):
            self.assertEqual(document[str(i)].binary, result[i].tobytes())

    def test_string(self):
        document = bson.SON(
            [("0", b"string_0"), ("1", b"str1"), ("2", b"utf8-2")])
        self.compare_results('<S10', document, document)
        document2 = bson.SON(
            [("0", b"st"), ("1", b"st"), ("2", b"ut")])
        self.compare_results('<S2', document, document2)
        self.compare_results('<S2', document2, document2)

    def test_binary(self):
        document = bson.SON(
            [("0", bson.Binary(b"binary_0")), ("1", bson.Binary(b"bin1")),
             ("2", bson.Binary(b"utf8-2"))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("<V15")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        for b in range(len(result)):
            # TODO: trailing null chars
            pass
            #self.assertEqual(str(document[str(b)]), str(result[b]))

    def test_datetime(self):
        document = bson.SON([("0", datetime.datetime(1970, 1, 1)),
                             ("1", datetime.datetime(1980, 1, 1)),
                             ("2", datetime.datetime(1990, 1, 1))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('int64')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        for b in range(len(result)):
            self.assertEqual(
                millis(document[str(b)] - datetime.datetime(1970, 1, 1)),
                result[b])

    def test_timestamp(self):
        document = bson.SON([("0",
                              bson.timestamp.Timestamp(time=00000, inc=77)),
                             ("1",
                              bson.timestamp.Timestamp(time=11111, inc=88)),
                             ("2",
                              bson.timestamp.Timestamp(time=22222, inc=99))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('uint64')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        data = [bson.timestamp.Timestamp(*struct.unpack("<ii", ts)) for ts in result]
        print("data=%s" % (data), )
        for b in range(len(result)):
            self.assertEqual(data[b], document[str(b)])

    def test_documents(self):
        document = bson.SON(
            [("0", bson.SON([("a", 1)])), ("1", bson.SON([("b", 2)])),
             ("2", bson.SON([("c", 3)]))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("<V12")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        for b in range(len(result)):
            doc = bson.BSON(result[b]).decode()
            self.assertEqual(dict(document[str(b)]), doc)

    def test_code(self):
        document = bson.SON([("0",
                              bson.code.Code("this is some code")),
                             ("1",
                              bson.code.Code("this is some more code")),
                             ("2",
                              bson.code.Code("this is some even more code"))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("<S35")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        for b in range(len(result)):
            self.assertEqual(str(document[str(b)]), result[b].decode('utf-8'))

    def test_regex(self):
        document = bson.SON([("0",
                              bson.regex.Regex('pattern', flags='i')),
                             ("1",
                              bson.regex.Regex('abcdef', flags='iiii')),
                             ("2",
                              bson.regex.Regex('123abc', flags='iiiiii'))])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("<S35")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        data = [r.split(b'\x00') for r in result]
        for b in range(len(document)):
            self.assertEqual(2, len(data[b]), "Bad regex=%s" % data[b])
            pattern, flags = data[b]
            self.assertEqual(bson.regex.Regex(pattern, str(flags)),
                             document[str(b)])

    def test_array(self):
        document = bson.SON([("0", 99),
                             ("1", 88),
                             ("2", 77)])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        print result
        print
        self.assertEqual(dtype, result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        # arrays of length 1 get automatically converted to constants, except if top-level array is 1.

        document = bson.SON([("0", [99,88,77])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        print result
        print
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON([("0", [[99,88,77]])])
        utf8 = bson._dict_to_bson(same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        print result
        print
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON([("0", [[99],[88],[77]])])
        utf8 = bson._dict_to_bson(same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        print result
        print
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        same_document = bson.SON([("0", [[[[[99]]]],[[[[[88]]]]],[[[[77]]]]])])
        utf8 = bson._dict_to_bson(same_document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("3int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        print result
        print
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        document = bson.SON([("0", [9,8]),
                             ("1", [6,5]),
                             ("2", [3,2])]) # [ [a,b], [c,d], [e,f] ]
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype("2int32")
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        document = bson.SON([("0", [[9,9],[8,8],[7,7]]),
                             ("1", [[6,6],[5,5],[4,4]]),
                             ("2", [[3,3],[2,2],[1,1]])])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('(3,2)int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))

        document = bson.SON([("0", [
                                    [[9],[9]],
                                    [[8],[8]],
                                    [[7],[7]]
                                   ]
                             ),
                             ("1", [
                                    [[6],[6]],
                                    [[5],[5]],
                                    [[4],[4]]
                                   ]
                             ),
                             ("2", [
                                    [[3],[3]],
                                    [[2],[2]],
                                    [[1],[1]]
                                   ]
                             )])
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('(3,2,1)int32')
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(dtype.subdtype[0], result.dtype)
        for b in range(len(result)):
            self.assertTrue(np.array_equal(document[str(b)], result[b]))


class TestCollection2Ndarray(unittest.TestCase):
    def test_iterator(self):
        client = pymongo.MongoClient()
        client.drop_database("bsonnumpy_test")
        client.bsonnumpy_test.coll.insert([{"x": i} for i in range(1000)])
        raw_coll = client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype("int32")
        ndarray = bsonnumpy.collection_to_ndarray((doc.raw for doc in cursor), dtype, raw_coll.count())
        # print "NDARRAY", ndarray

    def test_flexible_type(self):
        client = pymongo.MongoClient()
        client.drop_database("bsonnumpy_test")
        num_docs = 10
        names = [random.choice(string.lowercase)*10 for i in range(num_docs)]
        client.bsonnumpy_test.coll.insert([{"name": names[i],
                                            "grades": [random.random(),
                                                       random.random()]}
                                           for i in range(num_docs)])
        raw_coll = client.get_database(
            'bsonnumpy_test',
            codec_options=CodecOptions(document_class=RawBSONDocument)).coll
        cursor = raw_coll.find()

        dtype = np.dtype([('name', np.str_, 18), ('grades', np.float64, (2,))])
        ndarray = bsonnumpy.collection_to_ndarray((doc.raw for doc in cursor), dtype, raw_coll.count())
        print "NDARRAY", ndarray

        #~/Python-2.7.11/valgrind/coregrind/valgrind --tool=memcheck --leak-check=full
        # --suppressions=valgrind-python.supp python setup.

if __name__ == "__main__":
    unittest.main()
