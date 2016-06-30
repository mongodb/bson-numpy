import sys

if sys.version_info[:2] == (2, 6):
    import unittest2 as unittest
    from unittest2 import SkipTest
else:
    import unittest
    from unittest import SkipTest

PY3 = sys.version_info[0] >= 3

class BSONNumPyTestBase(unittest.TestCase):
    def assert_array_eq(self, actual, expected, err_msg=''):
        nptest.assert_array_equal(actual, expected, err_msg=err_msg)
        self.assertEqual(actual.dtype, expected.dtype)
