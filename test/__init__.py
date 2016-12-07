import sys

import bson
import numpy as np

import bsonnumpy

if sys.version_info[:2] == (2, 6):
    import unittest2 as unittest
    from unittest2 import SkipTest
else:
    import unittest
    from unittest import SkipTest

PY3 = sys.version_info[0] >= 3


class TestFromBSON(unittest.TestCase):
    def compare_results(self, np_type, document, compare_to):
        utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
        dtype = np.dtype(np_type)
        result = bsonnumpy.bson_to_ndarray(utf8, dtype)
        self.assertEqual(result.dtype, dtype)
        for i in range(len(result)):
            self.assertEqual(compare_to[str(i)], result[i],
                             "Comparison failed for type %s: %s != %s" % (
                                 dtype, compare_to[str(i)], result[i]))
