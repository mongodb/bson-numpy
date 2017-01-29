from datetime import datetime

import bson
import bsonnumpy
import numpy as np

from test import unittest

# See https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
int_codes = list('bhilqp')
uint_codes = list('BHILQP')
float_codes = list('efdg')
complex_codes = list('FDG')
numeric_codes = int_codes + uint_codes + float_codes + complex_codes
tinies = ['int8', 'uint8']
shorts = ['int16', 'uint16']
words = ['int32', 'uint32']
bytes_codes = ['S5', 'V10', 'U13']


class TestTypeErrors(unittest.TestCase):
    if hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesPattern = unittest.TestCase.assertRaisesRegex
    else:
        assertRaisesPattern = unittest.TestCase.assertRaisesRegexp

    def _test_error(self, value, bson_type_name, codes):
        data = bson._dict_to_bson({'x': value}, True,
                                  bson.DEFAULT_CODEC_OPTIONS)

        for code in codes:
            dtype = np.dtype([('x', code)])
            expected = "cannot convert %s to dtype" % bson_type_name

            with self.assertRaisesPattern(bsonnumpy.error, expected):
                bsonnumpy.sequence_to_ndarray(iter([data]), dtype, 1)

    def test_utf8(self):
        return self._test_error('foo', 'UTF-8 string', numeric_codes)

    def test_binary(self):
        return self._test_error(bson.Binary(b'foo'), 'Binary', numeric_codes)

    def test_objectid(self):
        return self._test_error(bson.ObjectId(), 'ObjectId',
                                numeric_codes + bytes_codes)

    def test_int32(self):
        return self._test_error(42, "Int32",
                                float_codes + complex_codes + bytes_codes
                                + tinies + shorts)

    def test_int64(self):
        return self._test_error(bson.Int64(42), "Int64",
                                float_codes + complex_codes + bytes_codes
                                + tinies + shorts + words)

    def test_double(self):
        return self._test_error(1.5, "Double",
                                int_codes + uint_codes + complex_codes
                                + bytes_codes)

    def test_datetime(self):
        return self._test_error(datetime.utcnow(), "Datetime",
                                float_codes + complex_codes + bytes_codes
                                + tinies + shorts + words)

    def test_no_fieldname(self):
        data = bson._dict_to_bson({'x': 1}, True, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype('i')
        with self.assertRaisesPattern(bsonnumpy.error,
                                      "must include field names"):
            bsonnumpy.sequence_to_ndarray([data], dtype, 1)
