import datetime
import struct

import bson
import bsonnumpy
import numpy as np

from test import TestFromBSON, millis, unittest


class TestToBSONScalars(unittest.TestCase):
    def test_integer32_types(self):
        array = np.array([99, 88, 77, 66], dtype=np.int32)
        bsonnumpy.ndarray_to_bson(array)

    def test_integer64_types(self):
        array = np.array([99, 88, 77, 66], dtype=np.int64)
        bsonnumpy.ndarray_to_bson(array)

    def test_bool(self):
        array = np.array([True, False, True, False], dtype=np.bool)
        bsonnumpy.ndarray_to_bson(array)

    def test_float64_types(self):
        array = np.array([99.99, 88.88, 77.77, 66.66], dtype=np.float64)
        bsonnumpy.ndarray_to_bson(array)

    # def test_oid(self):
    #     array = np.array([bson.ObjectId(), bson.ObjectId(),
    #                       bson.ObjectId(), bson.ObjectId()],
    #                      dtype=np.dtype('<V12'))
    #     bsonnumpy.ndarray_to_bson(array)

    def test_string(self):
        array = np.array([b"string_0", b"str1", b"utf8-2"],
                         dtype=np.dtype('<S2'))
        bsonnumpy.ndarray_to_bson(array)

    def test_binary(self):
        array = np.array([bson.Binary(b"binary_0"),
                          bson.Binary(b"bin1"),
                          bson.Binary(b"utf8-2")], dtype=np.dtype('<V15'))
        bsonnumpy.ndarray_to_bson(array)

    def test_subarray(self):
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         dtype=np.dtype('int32'))
        bsonnumpy.ndarray_to_bson(array)

        # def test_datetime(self):
        #     array = np.array([datetime.datetime(1970, 1, 1),
        #                       datetime.datetime(1970, 1, 2),
        #                       datetime.datetime(1970, 1, 3)],
        #                      dtype=np.dtype('int64'))
        #     bsonnumpy.ndarray_to_bson(array)
        #
        # def test_timestamp(self):
        #     array = np.array([
        #         bson.timestamp.Timestamp(time=00000, inc=77),
        #         bson.timestamp.Timestamp(time=00000, inc=88),
        #         bson.timestamp.Timestamp(time=00000, inc=99)],
        #         dtype=np.dtype('uint64'))
        #     bsonnumpy.ndarray_to_bson(array)
        #
        # def test_documents(self):
        #     array = np.array([
        #         bson.SON([("a", 1)]),
        #         bson.SON([("b", 2)]),
        #         bson.SON([("c", 3)])], dtype=np.dtype('<V35'))
        #     bsonnumpy.ndarray_to_bson(array)


class TestFromBSONScalars(TestFromBSON):
    def test_integer32_types(self):
        document = bson.SON([("0", 99), ("1", 88), ("2", 77), ("3", 66)])
        for np_type in [np.int32, np.uint32]:
            self.compare_results(np_type, document, document)

    def test_integer64_types(self):
        document = bson.SON(
            [("0", 99), ("1", 88), ("2", 77)])
        for np_type in [np.int_, np.intc, np.intp, np.uint64, np.int64]:
            self.compare_results(np_type, document, document)

    def test_bool(self):
        document = bson.SON([("0", True), ("1", False), ("2", True)])
        self.compare_results(np.bool_, document, document)

    def test_float64_types(self):
        document = bson.SON(
            [("0", 99.99), ("1", 88.88), ("2", 77.77)])
        for np_type in [np.float_, np.float64]:
            self.compare_results(np_type, document, document)

    def TODO_num_types(self):
        # np_types = [np.complex_, np.complex64, np.complex128, np.float32 ]
        # https://jira.mongodb.org/browse/SERVER-9342
        pass

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
            self.assertEqual(bytes(document['0'].ljust(15, b'\0')),
                             bytes(result[0]))

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


# Test all the unsupported types.
def _make_test_fn(value, type_name):
    def test(self):
        data = bson._dict_to_bson({"0": value},
                                  True,  # check_keys
                                  bson.DEFAULT_CODEC_OPTIONS)

        with self.assertRaises(bsonnumpy.error) as context:
            # dtype doesn't matter.
            bsonnumpy.bson_to_ndarray(data, np.dtype([('x', '<V99')]))

        self.assertIn("unsupported BSON type: %s" % type_name,
                      str(context.exception))

    return test


for value_, type_name_ in [
    (bson.Code(""), "Code"),
    (bson.Code("", {'scope': 1}), "Code with Scope"),
    ({}, "Sub-document"),
    (bson.MinKey(), "MinKey"),
    (bson.MaxKey(), "MaxKey"),
    (bson.regex.Regex("pattern"), "Regular Expression"),
    (bson.timestamp.Timestamp(0, 0), "Timestamp"),
    (None, "Null"),
]:
    test_name = "test_unsupported_%s" % type_name_
    setattr(TestFromBSONScalars, test_name, _make_test_fn(value_, type_name_))


if __name__ == "__main__":
    unittest.main()
