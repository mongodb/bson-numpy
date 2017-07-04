import datetime

import bson
import bsonnumpy
import numpy as np

from test import TestToNdarray, millis, unittest


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
