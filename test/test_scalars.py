import datetime

import bson
import bsonnumpy
import numpy as np

from test import TestToNdarray, millis, unittest


class TestToBSONScalars(unittest.TestCase):
    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_integer32_types(self):
        array = np.array([99, 88, 77, 66], dtype=np.int32)
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_integer64_types(self):
        array = np.array([99, 88, 77, 66], dtype=np.int64)
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_bool(self):
        array = np.array([True, False, True, False], dtype=np.bool)
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_float64_types(self):
        array = np.array([99.99, 88.88, 77.77, 66.66], dtype=np.float64)
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_oid(self):
        array = np.array([bson.ObjectId(), bson.ObjectId(),
                          bson.ObjectId(), bson.ObjectId()],
                         dtype=np.dtype('<V12'))
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_string(self):
        array = np.array([b"string_0", b"str1", b"utf8-2"],
                         dtype=np.dtype('<S2'))
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_binary(self):
        array = np.array([bson.Binary(b"binary_0"),
                          bson.Binary(b"bin1"),
                          bson.Binary(b"utf8-2")], dtype=np.dtype('<V15'))
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_subarray(self):
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         dtype=np.dtype('int32'))
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_datetime(self):
        array = np.array([datetime.datetime(1970, 1, 1),
                          datetime.datetime(1970, 1, 2),
                          datetime.datetime(1970, 1, 3)],
                         dtype=np.dtype('int64'))
        bsonnumpy.ndarray_to_sequence(array)

    @unittest.skip("TODO: these tests will be changed to test "
                   "ndarray_to_sequence for issue #5")
    def test_timestamp(self):
        array = np.array([
            bson.timestamp.Timestamp(time=00000, inc=77),
            bson.timestamp.Timestamp(time=00000, inc=88),
            bson.timestamp.Timestamp(time=00000, inc=99)],
            dtype=np.dtype('uint64'))
        bsonnumpy.ndarray_to_sequence(array)


class TestFromBSONScalars(TestToNdarray):
    def test_integer32_types(self):
        document = bson.SON([("a", 99)])
        for np_type in [np.int32, np.uint32]:
            self.compare_seq_to_ndarray_result(np.dtype([("a", np_type)]),
                                               document)

    def test_integer64_types(self):
        document = bson.SON([("a", 99), ("b", 88), ("foobar", 77)])
        for np_type in [np.int_, np.intc, np.intp, np.uint64, np.int64]:
            self.compare_seq_to_ndarray_result(
                np.dtype([("a", np_type), ("b", np_type), ("foobar", np_type)]),
                document)

    def test_float64_types(self):
        document = bson.SON([("a", 99.99)])
        for np_type in [np.float_, np.float64]:
            self.compare_seq_to_ndarray_result(
                np.dtype([("a", np_type)]), document)


# Test all the unsupported types.
def _make_test_fn(value, type_name, dtype):
    def test(self):
        data = bson._dict_to_bson({"a": value},
                                  True,  # check_keys
                                  bson.DEFAULT_CODEC_OPTIONS)

        with self.assertRaises(bsonnumpy.error) as context:
            bsonnumpy.sequence_to_ndarray([data], np.dtype([("a", dtype)]), 1)

        self.assertIn("unsupported BSON type: %s" % type_name,
                      str(context.exception))

    return test


for value_, type_name_, dtype_ in [
    (bson.Code(""), "Code", "V10"),
    (bson.Code("", {'scope': 1}), "Code with Scope", "V10"),
    ({}, "Sub-document", "V10"),
    (bson.MinKey(), "MinKey", "V10"),
    (bson.MaxKey(), "MaxKey", "V10"),
    (bson.regex.Regex("pattern"), "Regular Expression", "V10"),
    (bson.timestamp.Timestamp(0, 0), "Timestamp", "V10"),
    (None, "Null", "V10"),
]:
    test_name = "test_unsupported_%s" % type_name_.lower()
    setattr(TestFromBSONScalars, test_name,
            _make_test_fn(value_, type_name_, dtype_))

if __name__ == "__main__":
    unittest.main()
