import random
import string

import bson
import numpy as np

from test import unittest


class TestFlexibleTypes(unittest.TestCase):
    def test_flexible_type(self):
        num_dicts = 10
        lists = [(random.choice(string.ascii_lowercase) * 10,
                  [random.randint(0, 100),
                   random.randint(0, 100)]) for _ in range(num_dicts)]
        sons = bson.SON([
            (str(i),
             bson.SON([
                 ("name", lists[i][0]),
                 ("grades", lists[i][1])])) for i in range(num_dicts)
        ])

        dtype = np.dtype([('name', np.str, 18), ('grades', np.int32, (2,))])
        # Comment for now since erroring on 3.x
        # ndarray = bsonnumpy.bson_to_ndarray(utf8, dtype)

        ndarray = np.array(lists, dtype=dtype)

        self.assertEqual(ndarray.dtype, dtype)
        self.assertEqual(num_dicts, len(ndarray))
        for i in range(num_dicts):
            for desc in dtype.descr:
                name = desc[0]
                if len(desc) > 2:
                    self.assertTrue(
                        (sons[str(i)][name] == ndarray[i][name]).all())
                else:
                    self.assertEqual(sons[str(i)][name], ndarray[i][name])
